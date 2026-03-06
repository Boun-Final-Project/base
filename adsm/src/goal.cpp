#include "adsm/goal.h"

GoalNode::GoalNode() { }

GoalNode::GoalNode(int iteration, double x, double y, int type, double yaw)
    : iteration(static_cast<double>(iteration)), x(x), y(y), type(type), yaw(yaw) {}
