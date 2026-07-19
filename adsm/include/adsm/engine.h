#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace adsm_core {

struct Config {
  double k1 = 0.2;
  double random_sample_r = 3.0;
  int goal_cluster_num = 20;
  double obs_r = 0.2;
  double goal_reach_th = 0.5;
  double resample_time_th = 5.5;
  double gas_high_th = 0.3;
  double gas_low_th = 0.1;
  double sensor_window_length = 6.0;
  double frontier_search_th = 3.0;
  int rrt_max_iter = 200;
  double rrt_max_r = 3.0;
  double rrt_min_r = 0.70;
  double rrt_step_size = 0.3;
  double rrt_near_r = 0.5;
  double stuck_duration_th = 60.0;
};

struct Grid {
  int width = 0;
  int height = 0;
  double resolution = 0.0;
  double origin_x = 0.0;
  double origin_y = 0.0;
  std::vector<int8_t> data;  // -1 unknown, 0 free, 100 occupied

  bool valid() const;
  bool worldToMap(double wx, double wy, int &mx, int &my) const;
  void mapToWorld(int mx, int my, double &wx, double &wy) const;
  int cost(int mx, int my) const;
};

struct Input {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double gas = 0.0;
  double wind_speed = 0.0;
  double wind_direction = 0.0;
  double sim_time = 0.0;
  Grid map;
};

struct Decision {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  int goal_type = 2;  // 0 EPI, 1 EPR, 2 random
  double j = 0.0;
  double j_p = 0.0;
  double j_i = 0.0;
  int iteration = 0;
  bool gas_hit = false;
  bool resampled = false;
  std::size_t epi_size = 0;
  std::size_t epr_size = 0;
};

class Engine {
 public:
  explicit Engine(const Config &config = Config());
  void reset(uint32_t seed);
  Decision step(const Input &input);
  uint32_t seed() const { return seed_; }

 private:
  struct RRTNode {
    double x = 0.0;
    double y = 0.0;
    double cost = 0.0;
    int parent = -1;
  };
  struct Goal {
    int iteration = 0;
    double x = 0.0;
    double y = 0.0;
    int type = 2;
    double j = 0.0;
    double j_p = 0.0;
    double j_i = 0.0;
    double probability = 0.0;
    double frontier_size = 0.0;
  };

  Config cfg_;
  uint32_t seed_ = 0;
  int iteration_ = 0;
  bool initialized_ = false;
  bool gas_hit_ = false;
  bool do_sample_again_ = false;
  bool set_random_goal_ = false;
  double last_resample_time_ = 0.0;
  double stuck_x_ = std::numeric_limits<double>::quiet_NaN();
  double stuck_y_ = 0.0;
  double stuck_time_ = 0.0;
  Goal goal_;
  std::vector<std::pair<double, double>> gas_window_;
  std::vector<std::pair<double, double>> pose_history_;
  std::vector<Goal> epi_set_;
  std::vector<Goal> epr_set_;
  std::vector<RRTNode> rrt_nodes_;

  double randomUnit();
  double probability(const Input &in, double x, double y) const;
  bool reachedPoint(double x, double y) const;
  bool collisionFree(const Grid &map, double sx, double sy, double ex, double ey) const;
  std::vector<RRTNode> sampleRRT(const Grid &map, double sx, double sy);
  std::size_t frontierCount(const Grid &map, double sx, double sy) const;
  Goal safeRandomGoal(const Input &in, double sx, double sy);
  void classifyGas(const Input &in);
  void estimate(const Input &in, bool do_sample);
  Goal evaluate(const Input &in);
  void updateStuck(const Input &in);
};

}  // namespace adsm_core
