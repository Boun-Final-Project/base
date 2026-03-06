#pragma once

#include <cmath>
#include <random>

struct GoalNode {
    double iteration;
    double x;
    double y;
    int type;
    double yaw = 0.0;
    double j = 0.0;
    double j_p = 0.0; // probability
    double j_i = 0.0; // information
    double j_d = 0.0; // distance
    double probability = 0.0;
    double frontier_size = 0.0;
    double distance = 0.0;

    GoalNode();
    GoalNode(int iteration, double x, double y, int type, double yaw=0.0);
};
