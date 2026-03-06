#pragma once

#include "adsm/gridmap.h"
#include <vector>

struct RRTNode {
    int idx;
    int x_cell;
    int y_cell;
    double x;
    double y;
    int parent_idx;
    double cost;

    RRTNode(int idx, int x_cell, int y_cell, double x, double y, int parent_idx, double cost)
        : idx(idx), x_cell(x_cell), y_cell(y_cell), x(x), y(y), parent_idx(parent_idx), cost(cost) {}
};

class RRTSampler {
public:
    RRTSampler();
    RRTSampler(int max_iter, double sample_max_r, double obs_r, double step_size, double near_r=0.5);
    ~RRTSampler();
    std::vector<RRTNode*> sample (Gridmap* gridmap, double start_x, double start_y);

private:
    int max_iter_ = 200;
    double sample_max_r_ = 3.0;
    double step_size_ = 0.4;
    double obs_r_ = 0.2;
    double obs_check_step_ = 0.1;
    double near_r_ = 0.5;
    std::vector<RRTNode*> nodes_;

    double distance(double x1, double y1, double x2, double y2);
    RRTNode* find_nearest_node(double x, double y);
    std::vector<RRTNode*> find_near_nodes(double x, double y);
    void steer(double start_x, double start_y, double end_x, double end_y, double& new_node_x, double& new_node_y);
    bool check_collision(Gridmap* gridmap, double start_x, double start_y, double end_x, double end_y);
};
