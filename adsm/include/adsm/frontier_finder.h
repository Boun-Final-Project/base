#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include "adsm/gridmap.h"

class FrontierFinder {
public:
    FrontierFinder();
    FrontierFinder(double find_dis_th, double obs_r);
    std::vector<std::pair<double, double>> find(Gridmap* gridmap, double start_x, double start_y, bool only_first);

private:
    double find_dis_th_ = 2.0;
    double obs_r_ = 0.2;
    double obs_check_step_ = obs_r_;

    double distance(double x1, double y1, double x2, double y2);
};
