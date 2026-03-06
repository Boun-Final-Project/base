#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <olfaction_msgs/msg/gas_sensor.hpp>
#include <olfaction_msgs/msg/anemometer.hpp>
#include <gaden_msgs/srv/occupancy.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>
#include "adsm/frontier_finder.h"
#include "adsm/rrt_sampler.h"
#include "adsm/gridmap.h"
#include "adsm/goal.h"

double get_current_time();
std::string generate_uuid();
void save_vector_to_csv(const std::vector<std::vector<double>>& data, const std::string& filename, const std::string& header="");
void save_gridmap(std::vector<std::vector<int8_t>> data, const std::string& filename);

using NavigateToPose = nav2_msgs::action::NavigateToPose;

class Adsm : public rclcpp::Node {
public:
    Adsm();
    ~Adsm();
    void init();
    void loop();

    inline double distance(double x, double y) {
        return std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    }

    const int GOAL_RANDOM_TYPE = 2;
    const int GOAL_EPR_TYPE = 1;
    const int GOAL_EPI_TYPE = 0;

private:
    double k1_;
    double random_sample_r_;
    double rrt_max_r_;
    double rrt_min_r_;
    double frontier_search_th_;
    int goal_cluster_num_;
    double obs_r_;
    double goal_reach_th_;
    double resample_time_th_;
    double gas_max_ = 63000.0;
    double gas_low_th_ = 500.0;
    double gas_high_th_ = 3000.0;
    double sensor_window_len_ = 6.0;
    std::vector<std::pair<double, double>> gas_msg_queue_; // <time, concentration>

    int iter_ = 1;
    double iter_start_rostime_;
    double iter_rate_;
    int max_iter_;
    double source_x_;
    double source_y_;
    double source_th_;
    double stuck_th_;
    std::string random_run_id_;
    bool visual_;
    std::string data_path_;

    bool do_sample_ = false;
    bool do_sample_again_ = false;
    GoalNode goal_;
    // goal type: 0 epi, 1 epr, 2 random
    std::vector<GoalNode> goals_;
    std::vector<std::pair<double, double>> pose_history_;
    std::vector<GoalNode> epi_set_;
    std::vector<GoalNode> epr_set_;
    std::vector<RRTNode*> rrt_nodes_;
    Gridmap map_;
    RRTSampler rrt_sampler_;
    FrontierFinder frontier_finder_;
    double last_resample_time_;
    double cal_start_time_ = 0.0, cal_duration_ms_ = 0.0;
    double dis_to_source_ = 0.0;
    std::vector<double> stuck_info_{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}; // x, y, last_stuck_time
    double set_random_goal_ = false;
    std::string result_ = "";
    std::vector<std::vector<double>> info_log_;
    std::vector<std::vector<double>> targets_log_;
    std::vector<std::vector<double>> rrt_log_;
    std::vector<std::vector<double>> map_info_log_;
    std::vector<std::vector<int8_t>> map_log_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr real_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<olfaction_msgs::msg::GasSensor>::SharedPtr gas_sub_;
    rclcpp::Subscription<olfaction_msgs::msg::Anemometer>::SharedPtr anemometer_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr external_slam_map_sub_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visual_points_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visual_lines_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visual_text_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr slam_map_pub_;
    rclcpp::TimerBase::SharedPtr slam_map_timer_;
    bool slam_initialized_ = false;

    double x_ = std::numeric_limits<double>::quiet_NaN(), y_, z_;
    double roll_, pitch_, yaw_;
    double real_x_ = std::numeric_limits<double>::quiet_NaN(), real_y_, real_z_;
    double real_roll_, real_pitch_, real_yaw_;
    double gas_ = std::numeric_limits<double>::quiet_NaN();
    double gas_hit_ = false;
    double wind_speed_ = std::numeric_limits<double>::quiet_NaN();
    double wind_direction_ = std::numeric_limits<double>::quiet_NaN();

    // Distance / timing tracking
    double total_distance_ = 0.0;
    double prev_real_x_ = std::numeric_limits<double>::quiet_NaN();
    double prev_real_y_ = std::numeric_limits<double>::quiet_NaN();
    double start_rostime_ = 0.0;

    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void real_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void gas_sensor_callback(const olfaction_msgs::msg::GasSensor::SharedPtr msg);
    void anemometer_callback(const olfaction_msgs::msg::Anemometer::SharedPtr msg);
    void external_slam_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void publish_slam_map();

    double probability(double x, double y);
    bool reached_point(double x, double y);
    void create_random_gaol(double start_x, double start_y, double r, double& goal_x, double& goal_y);
    void observe();
    void estimate();
    void evaluate();
    void navigate();
    bool check_terminal();
    void visualize();
    void record_data();
    void save_data();
};
