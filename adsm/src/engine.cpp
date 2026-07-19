#include "adsm/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <stdexcept>

namespace adsm_core {
namespace {
constexpr int kUnknown = -1;
constexpr int kOccupied = 100;
constexpr int kEpi = 0;
constexpr int kEpr = 1;
constexpr int kRandom = 2;
double distance(double ax, double ay, double bx, double by) {
  return std::hypot(ax - bx, ay - by);
}
}  // namespace

bool Grid::valid() const {
  return width > 0 && height > 0 && resolution > 0.0 &&
         data.size() == static_cast<std::size_t>(width * height);
}

bool Grid::worldToMap(double wx, double wy, int &mx, int &my) const {
  if (!valid() || wx < origin_x || wy < origin_y) return false;
  mx = static_cast<int>((wx - origin_x) / resolution);
  my = static_cast<int>((wy - origin_y) / resolution);
  return mx >= 0 && my >= 0 && mx < width && my < height;
}

void Grid::mapToWorld(int mx, int my, double &wx, double &wy) const {
  wx = origin_x + (mx + 0.5) * resolution;
  wy = origin_y + (my + 0.5) * resolution;
}

int Grid::cost(int mx, int my) const {
  if (mx < 0 || my < 0 || mx >= width || my >= height) return kOccupied;
  return data[static_cast<std::size_t>(my * width + mx)];
}

Engine::Engine(const Config &config) : cfg_(config) { reset(0); }

void Engine::reset(uint32_t seed) {
  seed_ = seed;
  std::srand(seed_);
  iteration_ = 0;
  initialized_ = false;
  gas_hit_ = false;
  do_sample_again_ = false;
  set_random_goal_ = false;
  last_resample_time_ = 0.0;
  stuck_x_ = std::numeric_limits<double>::quiet_NaN();
  stuck_y_ = stuck_time_ = 0.0;
  goal_ = Goal{};
  gas_window_.clear();
  pose_history_.clear();
  epi_set_.clear();
  epr_set_.clear();
  rrt_nodes_.clear();
}

double Engine::randomUnit() {
  return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

void Engine::classifyGas(const Input &in) {
  gas_window_.push_back({in.sim_time, in.gas});
  while (!gas_window_.empty() &&
         in.sim_time - gas_window_.front().first > cfg_.sensor_window_length) {
    gas_window_.erase(gas_window_.begin());
  }
  if (in.gas < cfg_.gas_low_th) {
    gas_hit_ = false;
  } else if (in.gas > cfg_.gas_high_th) {
    gas_hit_ = true;
  } else {
    bool decreasing = true;
    for (std::size_t i = 1; i < gas_window_.size(); ++i) {
      if (gas_window_[i].second - gas_window_[i - 1].second >= 0.0 ||
          gas_window_[i - 1].second - cfg_.gas_high_th >= 0.0) {
        decreasing = false;
      }
    }
    gas_hit_ = !decreasing;
  }
}

double Engine::probability(const Input &in, double x, double y) const {
  constexpr double Q = 4.0, D = 1.0, tau = 250.0;
  const double V = in.wind_speed;
  const double phi = in.yaw - in.wind_direction;
  const double lambda = std::sqrt(D * tau / (1.0 + V * V * tau / (4.0 * D)));
  const double d = distance(x, y, in.x, in.y);
  const double dx = in.x - x, dy = in.y - y;
  return Q / (4.0 * M_PI * D * std::abs(d + 0.0001)) * std::exp(-d / lambda) *
         std::exp(-dx * V * std::cos(phi) / (2.0 * D)) *
         std::exp(-dy * V * std::sin(phi) / (2.0 * D));
}

bool Engine::reachedPoint(double x, double y) const {
  for (const auto &p : pose_history_) {
    if (distance(p.first, p.second, x, y) < cfg_.goal_reach_th) return true;
  }
  return false;
}

bool Engine::collisionFree(const Grid &map, double sx, double sy, double ex, double ey) const {
  const int obs_cells = static_cast<int>(cfg_.obs_r / map.resolution);
  const double check_step = cfg_.obs_r / 2.0;
  const double d = distance(sx, sy, ex, ey);
  const double theta = std::atan2(ey - sy, ex - sx);
  const int n = static_cast<int>(std::floor(d / check_step));
  for (int p = 0; p <= n + 1; ++p) {
    const double step = p <= n ? p * check_step : d;
    int mx, my;
    if (!map.worldToMap(sx + step * std::cos(theta), sy + step * std::sin(theta), mx, my))
      return false;
    for (int x = std::max(0, mx - obs_cells); x < std::min(map.width, mx + obs_cells); ++x) {
      for (int y = std::max(0, my - obs_cells); y < std::min(map.height, my + obs_cells); ++y) {
        if (map.cost(x, y) == kUnknown || map.cost(x, y) >= kOccupied) return false;
      }
    }
  }
  return true;
}

std::vector<Engine::RRTNode> Engine::sampleRRT(const Grid &map, double sx, double sy) {
  std::vector<RRTNode> nodes{{sx, sy, 0.0, -1}};
  for (int iter = 0; iter < cfg_.rrt_max_iter; ++iter) {
    const double theta = randomUnit() * 2.0 * M_PI;
    const double radius = randomUnit() * (cfg_.rrt_max_r - cfg_.rrt_step_size) + cfg_.rrt_step_size;
    const double rx = sx + radius * std::cos(theta), ry = sy + radius * std::sin(theta);
    int nearest = 0;
    double nearest_d = distance(nodes[0].x, nodes[0].y, rx, ry);
    for (std::size_t i = 1; i < nodes.size(); ++i) {
      const double d = distance(nodes[i].x, nodes[i].y, rx, ry);
      if (d < nearest_d) { nearest = static_cast<int>(i); nearest_d = d; }
    }
    const double a = std::atan2(ry - nodes[nearest].y, rx - nodes[nearest].x);
    double nx = nodes[nearest].x + cfg_.rrt_step_size * std::cos(a);
    double ny = nodes[nearest].y + cfg_.rrt_step_size * std::sin(a);
    if (distance(nx, ny, rx, ry) < cfg_.rrt_step_size) { nx = rx; ny = ry; }
    int mx, my;
    if (!map.worldToMap(nx, ny, mx, my) || !collisionFree(map, nodes[nearest].x, nodes[nearest].y, nx, ny))
      continue;
    RRTNode node{nx, ny, nodes[nearest].cost + distance(nx, ny, nodes[nearest].x, nodes[nearest].y), nearest};
    std::vector<int> near;
    for (std::size_t i = 0; i < nodes.size(); ++i)
      if (distance(nodes[i].x, nodes[i].y, nx, ny) < cfg_.rrt_near_r) near.push_back(static_cast<int>(i));
    int parent = -1;
    double best = node.cost;
    for (int i : near) {
      const double c = nodes[i].cost + distance(nodes[i].x, nodes[i].y, nx, ny);
      if (c < best) { best = c; parent = i; }
    }
    if (parent >= 0) {
      const double pa = std::atan2(ny - nodes[parent].y, nx - nodes[parent].x);
      double tx = nodes[parent].x + cfg_.rrt_step_size * std::cos(pa);
      double ty = nodes[parent].y + cfg_.rrt_step_size * std::sin(pa);
      if (distance(tx, ty, nx, ny) < cfg_.rrt_step_size) { tx = nx; ty = ny; }
      if (collisionFree(map, nodes[parent].x, nodes[parent].y, tx, ty)) {
        node = {tx, ty, nodes[parent].cost + distance(nodes[parent].x, nodes[parent].y, tx, ty), parent};
      }
    }
    const int new_idx = static_cast<int>(nodes.size());
    for (int i : near) {
      const double c = node.cost + distance(nodes[i].x, nodes[i].y, node.x, node.y);
      if (c < nodes[i].cost) { nodes[i].cost = c; nodes[i].parent = new_idx; }
    }
    nodes.push_back(node);
  }
  return nodes;
}

std::size_t Engine::frontierCount(const Grid &map, double sx, double sy) const {
  int start_x, start_y;
  if (!map.worldToMap(sx, sy, start_x, start_y)) return 0;
  const int obs_cells = static_cast<int>(cfg_.obs_r / map.resolution);
  std::queue<std::pair<int, int>> q;
  std::vector<uint8_t> visited(static_cast<std::size_t>(map.width * map.height), 0);
  std::vector<int> path_cost(static_cast<std::size_t>(map.width * map.height), 0);
  auto idx = [&map](int x, int y) { return static_cast<std::size_t>(y * map.width + x); };
  q.push({start_x, start_y}); visited[idx(start_x, start_y)] = 1;
  std::size_t count = 0;
  const int dx[4] = {0, 1, 0, -1}, dy[4] = {1, 0, -1, 0};
  while (!q.empty()) {
    auto [x, y] = q.front(); q.pop();
    const int c = map.cost(x, y);
    if (c == kUnknown || c >= kOccupied) continue;
    bool obstacle = false;
    for (int xx = std::max(0, x - obs_cells); xx < std::min(map.width, x + obs_cells); ++xx)
      for (int yy = std::max(0, y - obs_cells); yy < std::min(map.height, y + obs_cells); ++yy)
        obstacle = obstacle || map.cost(xx, yy) >= kOccupied;
    if (obstacle) continue;
    for (int k = 0; k < 4; ++k) {
      const int nx = x + dx[k], ny = y + dy[k];
      if (nx < 0 || ny < 0 || nx >= map.width || ny >= map.height || visited[idx(nx, ny)]) continue;
      visited[idx(nx, ny)] = 1;
      path_cost[idx(nx, ny)] = path_cost[idx(x, y)] + 1;
      if (map.resolution * path_cost[idx(nx, ny)] > cfg_.frontier_search_th) continue;
      const int nc = map.cost(nx, ny);
      if (nc == kUnknown) ++count;
      else if (nc < kOccupied) q.push({nx, ny});
    }
  }
  return count;
}

Engine::Goal Engine::safeRandomGoal(const Input &in, double sx, double sy) {
  for (int tries = 0; tries < 200; ++tries) {
    const double theta = randomUnit() * 2.0 * M_PI;
    const double radius = randomUnit() * cfg_.random_sample_r;
    const double x = sx + radius * std::cos(theta), y = sy + radius * std::sin(theta);
    int mx, my;
    if (in.map.worldToMap(x, y, mx, my) && in.map.cost(mx, my) == 0)
      return Goal{iteration_, x, y, kRandom};
  }
  return Goal{iteration_, in.x, in.y, kRandom};
}

void Engine::estimate(const Input &in, bool do_sample) {
  if (do_sample) {
    std::vector<RRTNode> sampled;
    for (int attempt = 0; attempt < 10; ++attempt) {
      double sx = in.x, sy = in.y;
      if (attempt > 0) {
        Goal seed = safeRandomGoal(in, in.x, in.y);
        sx = seed.x; sy = seed.y;
      }
      sampled = sampleRRT(in.map, sx, sy);
      if (sampled.size() >= 2) break;
      sampled.clear();
    }
    if (sampled.size() >= 2) {
      epi_set_.clear();
      rrt_nodes_ = std::move(sampled);
    }
  }
  do_sample_again_ = rrt_nodes_.size() < 2;
  if (do_sample && !do_sample_again_) {
    std::vector<int> categories(rrt_nodes_.size(), -1), farthest(cfg_.goal_cluster_num, -1);
    std::vector<double> farthest_d(cfg_.goal_cluster_num, -1.0);
    for (std::size_t i = 1; i < rrt_nodes_.size(); ++i) {
      double a = std::atan2(rrt_nodes_[i].y - in.y, rrt_nodes_[i].x - in.x);
      if (a < 0.0) a += 2.0 * M_PI;
      int cat = std::min(cfg_.goal_cluster_num - 1,
                         static_cast<int>(a / (2.0 * M_PI) * cfg_.goal_cluster_num));
      categories[i] = cat;
      const double d = distance(rrt_nodes_[i].x, rrt_nodes_[i].y, in.x, in.y);
      if (d > farthest_d[cat]) { farthest_d[cat] = d; farthest[cat] = static_cast<int>(i); }
    }
    std::vector<uint8_t> is_farthest(rrt_nodes_.size(), 0);
    for (int i : farthest) if (i >= 0) is_farthest[static_cast<std::size_t>(i)] = 1;
    epi_set_.clear();
    for (std::size_t i = 1; i < rrt_nodes_.size(); ++i) {
      if (categories[i] < 0) continue;
      if (!is_farthest[i]) { epi_set_.push_back({iteration_, rrt_nodes_[i].x, rrt_nodes_[i].y, kEpi}); continue; }
      bool duplicate = false;
      for (const auto &g : epr_set_)
        if (distance(g.x, g.y, rrt_nodes_[i].x, rrt_nodes_[i].y) < cfg_.goal_reach_th / 2.0) duplicate = true;
      const auto nfrontier = duplicate ? 0 : frontierCount(in.map, rrt_nodes_[i].x, rrt_nodes_[i].y);
      if (duplicate || nfrontier == 0) epi_set_.push_back({iteration_, rrt_nodes_[i].x, rrt_nodes_[i].y, kEpi});
      else { Goal g{iteration_, rrt_nodes_[i].x, rrt_nodes_[i].y, kEpr}; g.frontier_size = nfrontier; epr_set_.push_back(g); }
    }
  }
  epi_set_.erase(std::remove_if(epi_set_.begin(), epi_set_.end(), [&](const Goal &g) {
    return distance(g.x, g.y, in.x, in.y) < cfg_.rrt_min_r;
  }), epi_set_.end());
  for (auto &g : epi_set_) { g.probability = gas_hit_ ? probability(in, g.x, g.y) : 0.0; g.frontier_size = 0.0; }
  epr_set_.erase(std::remove_if(epr_set_.begin(), epr_set_.end(), [&](Goal &g) {
    if (distance(g.x, g.y, in.x, in.y) < cfg_.rrt_max_r + cfg_.frontier_search_th)
      g.frontier_size = static_cast<double>(frontierCount(in.map, g.x, g.y));
    return reachedPoint(g.x, g.y) || g.frontier_size == 0.0;
  }), epr_set_.end());
  for (auto &g : epr_set_) g.probability = gas_hit_ ? probability(in, g.x, g.y) : 0.0;
}

Engine::Goal Engine::evaluate(const Input &in) {
  std::vector<Goal> goals = epi_set_;
  goals.insert(goals.end(), epr_set_.begin(), epr_set_.end());
  if (goals.empty() || set_random_goal_) return safeRandomGoal(in, goal_.x, goal_.y);
  double sum_p = 0.0, sum_i = 0.0;
  for (const auto &g : goals) { sum_p += g.probability; sum_i += g.frontier_size; }
  for (auto &g : goals) {
    g.j_p = sum_p > 0.0 ? g.probability / sum_p : 0.0;
    g.j_i = sum_i > 0.0 ? g.frontier_size / sum_i : 0.0;
    const double f = static_cast<double>(g.iteration) / static_cast<double>(iteration_);
    g.j = g.j_p + cfg_.k1 * f * g.j_i;
  }
  return *std::max_element(goals.begin(), goals.end(), [](const Goal &a, const Goal &b) { return a.j < b.j; });
}

void Engine::updateStuck(const Input &in) {
  if (std::isnan(stuck_x_)) { stuck_x_ = in.x; stuck_y_ = in.y; stuck_time_ = in.sim_time; }
  else if (distance(stuck_x_, stuck_y_, in.x, in.y) > 0.15) {
    stuck_x_ = in.x; stuck_y_ = in.y; stuck_time_ = in.sim_time;
  }
  set_random_goal_ = in.sim_time - stuck_time_ > cfg_.stuck_duration_th / 3.0;
}

Decision Engine::step(const Input &in) {
  if (!in.map.valid()) throw std::invalid_argument("ADSM requires a non-empty occupancy grid");
  if (!initialized_) {
    initialized_ = true;
    goal_ = Goal{0, in.x, in.y, kRandom};
    last_resample_time_ = in.sim_time;
    stuck_time_ = in.sim_time;
  }
  ++iteration_;
  classifyGas(in);
  pose_history_.push_back({in.x, in.y});
  bool do_sample = distance(in.x, in.y, goal_.x, goal_.y) < cfg_.goal_reach_th ||
                   in.sim_time - last_resample_time_ > cfg_.resample_time_th || do_sample_again_;
  if (do_sample) last_resample_time_ = in.sim_time;
  estimate(in, do_sample);
  goal_ = evaluate(in);
  updateStuck(in);  // original check_terminal affects the following iteration
  return Decision{goal_.x, goal_.y, 0.0, goal_.type, goal_.j, goal_.j_p, goal_.j_i,
                  iteration_, gas_hit_, do_sample, epi_set_.size(), epr_set_.size()};
}

}  // namespace adsm_core
