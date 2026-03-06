#include "adsm/gridmap.h"

Gridmap::Gridmap() {}

void Gridmap::update(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    is_grid_set_ = true;
    occupancy_grid_ = *msg;
    frame_id_ = msg->header.frame_id;
    size_x_ = msg->info.width;
    size_y_ = msg->info.height;
    resolution_ = msg->info.resolution;
    origin_x_ = msg->info.origin.position.x;
    origin_y_ = msg->info.origin.position.y;
}

void Gridmap::initFromReference(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    is_grid_set_ = true;
    occupancy_grid_ = *msg;
    frame_id_ = msg->header.frame_id;
    size_x_ = msg->info.width;
    size_y_ = msg->info.height;
    resolution_ = msg->info.resolution;
    origin_x_ = msg->info.origin.position.x;
    origin_y_ = msg->info.origin.position.y;
    // Fill all cells with unknown (-1)
    std::fill(occupancy_grid_.data.begin(), occupancy_grid_.data.end(), -1);
}

void Gridmap::initEmpty(int width, int height, double resolution, double origin_x, double origin_y) {
    is_grid_set_ = true;
    frame_id_ = "map";
    size_x_ = width;
    size_y_ = height;
    resolution_ = resolution;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    occupancy_grid_.header.frame_id = frame_id_;
    occupancy_grid_.info.width = width;
    occupancy_grid_.info.height = height;
    occupancy_grid_.info.resolution = resolution;
    occupancy_grid_.info.origin.position.x = origin_x;
    occupancy_grid_.info.origin.position.y = origin_y;
    occupancy_grid_.info.origin.orientation.w = 1.0;
    occupancy_grid_.data.assign(width * height, -1);
}

void Gridmap::setOutletMask(const std::vector<bool>& mask) {
    outlet_mask_ = mask;
    has_outlet_mask_ = true;
}

bool Gridmap::hasOutlet(int mx, int my) const {
    if (!has_outlet_mask_) return false;
    int idx = getIndex(mx, my);
    if (idx < 0 || idx >= static_cast<int>(outlet_mask_.size())) return false;
    return outlet_mask_[idx];
}

void Gridmap::setCost(int mx, int my, int8_t value) {
    occupancy_grid_.data[getIndex(mx, my)] = value;
}

bool Gridmap::worldToMap(double wx, double wy, int &mx, int &my) const {
    if (wx < origin_x_ || wy < origin_y_)
        return false;

    mx = (int)((wx - origin_x_) / resolution_);
    my = (int)((wy - origin_y_) / resolution_);

    if (mx < size_x_ && my < size_y_)
        return true;

    return false;
}

void Gridmap::mapToWorld(int mx, int my, double &wx, double &wy) const
{
    wx = origin_x_ + (mx + 0.5) * resolution_;
    wy = origin_y_ + (my + 0.5) * resolution_;
}

std::string Gridmap::getFrameID() const {
    return frame_id_;
}

int Gridmap::getSizeInCellsX() const
{
    return size_x_;
}

int Gridmap::getSizeInCellsY() const
{
    return size_y_;
}

double Gridmap::getOriginX() const
{
    return origin_x_;
}

double Gridmap::getOriginY() const
{
    return origin_y_;
}

double Gridmap::getResolution() const
{
    return resolution_;
}

int Gridmap::getCost(int mx, int my) const
{
    return occupancy_grid_.data[getIndex(mx, my)];
}

std::vector<int8_t> Gridmap::getData() const {
    std::vector<int8_t> map_data;
    map_data.reserve(occupancy_grid_.data.size());
    map_data.insert(map_data.end(), occupancy_grid_.data.begin(), occupancy_grid_.data.end());
    return map_data;
}

double Gridmap::is_grid_set() const
{
    return is_grid_set_;
}

nav_msgs::msg::OccupancyGrid Gridmap::getOccupancyGrid() const {
    return occupancy_grid_;
}
