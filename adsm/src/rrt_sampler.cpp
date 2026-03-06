#include "adsm/rrt_sampler.h"
#include <cmath>
#include <limits>

RRTSampler::RRTSampler() {}

RRTSampler::RRTSampler(int max_iter, double sample_max_r, double obs_r, double step_size, double near_r)
    : max_iter_(max_iter), sample_max_r_(sample_max_r), step_size_(step_size), obs_r_(obs_r), near_r_(near_r) {
    obs_check_step_ = obs_r_ / 2.0;
}

double RRTSampler::distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

RRTNode* RRTSampler::find_nearest_node(double x, double y) {
    RRTNode* nearest = nullptr;
    double min_dist = std::numeric_limits<double>::max();
    for (auto& node : nodes_) {
        double dist = distance(x, y, node->x, node->y);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = node;
        }
    }
    return nearest;
}

std::vector<RRTNode*> RRTSampler::find_near_nodes(double x, double y) {
    std::vector<RRTNode*> nodes;
    for (auto& node : nodes_) {
        double dist = distance(x, y, node->x, node->y);
        if (dist < near_r_) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

void RRTSampler::steer(double start_x, double start_y, double end_x, double end_y, double& new_node_x, double& new_node_y) {
    double theta = atan2(end_y - start_y, end_x - start_x);
    new_node_x = start_x + step_size_ * cos(theta);
    new_node_y = start_y + step_size_ * sin(theta);
    double dis = distance(new_node_x, new_node_y, end_x, end_y);
    if (dis < step_size_) {
        new_node_x = end_x;
        new_node_y = end_y;
    }
}

bool RRTSampler::check_collision(Gridmap* gridmap, double start_x, double start_y, double end_x, double end_y){
    int obs_r = (int)(obs_r_ / gridmap->getResolution());
    double theta = atan2(end_y - start_y, end_x - start_x);
    std::vector<std::pair<double, double>> path;

    double n_expand = static_cast<int>(floor(distance(start_x, start_y, end_x, end_y) / obs_check_step_));
    for (int i = 0; i <= n_expand; ++i) {
        path.push_back({start_x + i * obs_check_step_ * cos(theta), start_y + i * obs_check_step_ * sin(theta)});
    }
    path.push_back({end_x, end_y});

    int width = gridmap->getSizeInCellsX();
    int height = gridmap->getSizeInCellsY();
    for (size_t i = 0; i < path.size(); ++i) {
        int x, y;
        if (!gridmap->worldToMap(path[i].first, path[i].second, x, y)) {
            continue;
        }
        int x_lim_s = std::max(0, static_cast<int>(x) - obs_r);
        int x_lim_e = std::min(static_cast<int>(width), static_cast<int>(x) + obs_r);
        int y_lim_s = std::max(0, static_cast<int>(y) - obs_r);
        int y_lim_e = std::min(static_cast<int>(height), static_cast<int>(y) + obs_r);
        for (int xi = x_lim_s; xi < x_lim_e; ++xi) {
            for (int yi = y_lim_s; yi < y_lim_e; ++yi) {
                if (gridmap->getCost(xi, yi) == Gridmap::NO_INFORMATION || gridmap->getCost(xi, yi) == Gridmap::LETHAL_OBSTACLE) {
                    return true;
                }
            }
        }
    }

    return false;
}

std::vector<RRTNode*> RRTSampler::sample (Gridmap* gridmap, double start_x_w, double start_y_w) {
    std::lock_guard<std::mutex> lock(gridmap->getMutex());

    int width = gridmap->getSizeInCellsX();
    int height = gridmap->getSizeInCellsY();
    (void)width;
    (void)height;

    // start node
    nodes_.clear();
    int start_x_m, start_y_m;
    gridmap->worldToMap(start_x_w, start_y_w, start_x_m, start_y_m);
    RRTNode* start_node = new RRTNode(0, start_x_m, start_y_m, start_x_w, start_y_w, -1, 0.0);
    nodes_.push_back(start_node);

    for (int i = 0; i < max_iter_; ++i) {
        // create a random node
        double rand_theta = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
        double rand_r = static_cast<double>(rand()) / RAND_MAX * (sample_max_r_ - step_size_) + step_size_;
        double rand_x_w = start_x_w + rand_r * cos(rand_theta);
        double rand_y_w = start_y_w + rand_r * sin(rand_theta);
        // find nearest node
        RRTNode* nearest_node = find_nearest_node(rand_x_w, rand_y_w);
        double new_x_w, new_y_w;
        steer(nearest_node->x, nearest_node->y, rand_x_w, rand_y_w, new_x_w, new_y_w);
        int new_x_m, new_y_m;
        if (!gridmap->worldToMap(new_x_w, new_y_w, new_x_m, new_y_m)) {
            continue;
        }
        // create new node & assign parent
        double new_cost = nearest_node->cost + distance(new_x_w, new_y_w, nearest_node->x, nearest_node->y);
        RRTNode* new_node = new RRTNode(nodes_.back()->idx + 1, new_x_m, new_y_m, new_x_w, new_y_w, nearest_node->idx, new_cost);

        if (!check_collision(gridmap, nearest_node->x, nearest_node->y, new_node->x, new_node->y)) {
            std::vector<RRTNode*> near_nodes = find_near_nodes(new_node->x, new_node->y);
            if (near_nodes.size() > 0) {
                // find new parent
                RRTNode* new_parent = nullptr;
                double new_cost = new_node->cost;
                for (auto& node : near_nodes) {
                    double cost_t = node->cost + distance(node->x, node->y, new_node->x, new_node->y);
                    if (cost_t < new_cost) {
                        new_cost = cost_t;
                        new_parent = node;
                    }
                }
                if (!(new_parent == nullptr)) {
                    double new_x_t_w, new_y_t_w;
                    steer(new_parent->x, new_parent->y, new_node->x, new_node->y, new_x_t_w, new_y_t_w);
                    if (!check_collision(gridmap, new_parent->x, new_parent->y, new_x_t_w, new_y_t_w)) {
                        int new_x_t_m, new_y_t_m;
                        if (gridmap->worldToMap(new_x_t_w, new_y_t_w, new_x_t_m, new_y_t_m)) {
                            delete new_node;
                            new_node = new RRTNode(nodes_.back()->idx + 1, new_x_t_m, new_y_t_m, new_x_t_w, new_y_t_w, new_parent->idx, new_cost);
                        }
                    }
                }
                // rewire
                for (auto&node : near_nodes) {
                    double cost_t = new_node->cost + distance(node->x, node->y, new_node->x, new_node->y);
                    if (cost_t < node->cost) {
                        node->cost = cost_t;
                        node->parent_idx = new_node->idx;
                    }
                }
            }
            nodes_.push_back(new_node);
        }

    }

    return nodes_;
}

RRTSampler::~RRTSampler() {
    nodes_.clear();
}
