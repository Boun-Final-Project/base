#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <vector>
#include <mutex>

class Gridmap {
public:
    static const int NO_INFORMATION = -1;
    static const int LETHAL_OBSTACLE = 100;

    Gridmap();
    void update(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void initFromReference(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void initEmpty(int width, int height, double resolution, double origin_x, double origin_y);
    void setOutletMask(const std::vector<bool>& mask);
    bool hasOutlet(int mx, int my) const;
    void setCost(int mx, int my, int8_t value);

    inline int getIndex(int mx, int my) const{
        return my * size_x_ + mx;
    }
    bool worldToMap(double wx, double wy, int &mx, int &my) const;
    void mapToWorld(int mx, int my, double &wx, double &wy) const;
    std::string getFrameID() const;
    int getSizeInCellsX() const;
    int getSizeInCellsY() const;
    double getOriginX() const;
    double getOriginY() const;
    double getResolution() const;
    int getCost(int mx, int my) const;
    std::vector<int8_t> getData() const;
    double is_grid_set() const;
    nav_msgs::msg::OccupancyGrid getOccupancyGrid() const;

    std::mutex& getMutex() const { return mutex_; }
    void lock() const { mutex_.lock(); }
    void unlock() const { mutex_.unlock(); }

private:
    nav_msgs::msg::OccupancyGrid occupancy_grid_;
    std::string frame_id_ = "";
    int size_x_ = 0;
    int size_y_ = 0;
    double resolution_ = 0.0;
    double origin_x_ = 0.0;
    double origin_y_ = 0.0;
    bool is_grid_set_ = false;
    std::vector<bool> outlet_mask_;
    bool has_outlet_mask_ = false;
    mutable std::mutex mutex_;
};
