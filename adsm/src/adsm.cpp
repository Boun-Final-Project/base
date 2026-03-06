#include "adsm/adsm.h"

double get_current_time() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

std::string generate_uuid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    const char* hex = "0123456789abcdef";
    std::string uuid;
    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) uuid += '-';
        uuid += hex[dis(gen)];
    }
    return uuid;
}

Adsm::Adsm() : Node("adsm_node") {
    RCLCPP_INFO(this->get_logger(), "Read parameters.");

    // Algorithm parameters
    this->declare_parameter("k1", 0.2);
    this->declare_parameter("random_sample_r", 3.0);
    this->declare_parameter("goal_cluster_num", 20);
    this->declare_parameter("obs_r", 0.2);
    this->declare_parameter("goal_reach_th", 0.5);
    this->declare_parameter("resample_time_th", 5.5);
    this->declare_parameter("gas_max", 10.0);
    this->declare_parameter("gas_high_th", 0.3);
    this->declare_parameter("gas_low_th", 0.1);
    this->declare_parameter("sensor_window_length", 6.0);

    k1_ = this->get_parameter("k1").as_double();
    random_sample_r_ = this->get_parameter("random_sample_r").as_double();
    goal_cluster_num_ = this->get_parameter("goal_cluster_num").as_int();
    obs_r_ = this->get_parameter("obs_r").as_double();
    goal_reach_th_ = this->get_parameter("goal_reach_th").as_double();
    resample_time_th_ = this->get_parameter("resample_time_th").as_double();
    gas_max_ = this->get_parameter("gas_max").as_double();
    gas_high_th_ = this->get_parameter("gas_high_th").as_double();
    gas_low_th_ = this->get_parameter("gas_low_th").as_double();
    sensor_window_len_ = this->get_parameter("sensor_window_length").as_double();
    RCLCPP_INFO(this->get_logger(), "k1 %.2f", k1_);
    RCLCPP_INFO(this->get_logger(), "gas_max %.2f, gas_high_th %.2f, gas_low_th %.2f", gas_max_, gas_high_th_, gas_low_th_);
    RCLCPP_INFO(this->get_logger(), "sensor_window_length %.2f", sensor_window_len_);

    // Run parameters
    this->declare_parameter("iter_rate", 1.0);
    this->declare_parameter("max_iter", 360);
    this->declare_parameter("source_x", 0.0);
    this->declare_parameter("source_y", 0.0);
    this->declare_parameter("source_th", 0.5);
    this->declare_parameter("stuck_duration_th", 60.0);
    this->declare_parameter("visual", true);
    this->declare_parameter("data_path", "/tmp/adsm_results");

    iter_rate_ = this->get_parameter("iter_rate").as_double();
    max_iter_ = this->get_parameter("max_iter").as_int();
    source_x_ = this->get_parameter("source_x").as_double();
    source_y_ = this->get_parameter("source_y").as_double();
    source_th_ = this->get_parameter("source_th").as_double();
    stuck_th_ = this->get_parameter("stuck_duration_th").as_double();
    visual_ = this->get_parameter("visual").as_bool();
    data_path_ = this->get_parameter("data_path").as_string();
    random_run_id_ = generate_uuid();
    data_path_ = data_path_ + '/' + random_run_id_;
    RCLCPP_INFO(this->get_logger(), "iter_rate %.2f, max_iter %d", iter_rate_, max_iter_);
    RCLCPP_INFO(this->get_logger(), "source_x %.2f, source_y %.2f", source_x_, source_y_);

    // Topic parameters
    this->declare_parameter("pose_topic", "/PioneerP3DX/odom");
    this->declare_parameter("real_pose_topic", "/PioneerP3DX/ground_truth");
    this->declare_parameter("laser_topic", "/PioneerP3DX/laser_scanner");
    this->declare_parameter("gas_sensor_topic", "/fake_pid/Sensor_reading");
    this->declare_parameter("anemometer_topic", "/fake_anemometer/WindSensor_reading");
    this->declare_parameter("nav_action", "/PioneerP3DX/navigate_to_pose");
    this->declare_parameter("gaden_occupancy_service", "/gaden_environment/occupancyMap3D");
    this->declare_parameter("z_level", 5);
    this->declare_parameter("external_slam_map_topic", "/slam_node/slam_map");

    std::string pose_topic = this->get_parameter("pose_topic").as_string();
    std::string real_pose_topic = this->get_parameter("real_pose_topic").as_string();
    std::string laser_topic = this->get_parameter("laser_topic").as_string();
    std::string gas_sensor_topic = this->get_parameter("gas_sensor_topic").as_string();
    std::string anemometer_topic = this->get_parameter("anemometer_topic").as_string();
    std::string nav_action = this->get_parameter("nav_action").as_string();
    std::string external_slam_map_topic = this->get_parameter("external_slam_map_topic").as_string();

    // RRT parameters
    this->declare_parameter("frontier_search_th", 3.0);
    this->declare_parameter("rrt_max_iter", 200);
    this->declare_parameter("rrt_max_r", 3.0);
    this->declare_parameter("rrt_min_r", 0.70);
    this->declare_parameter("rrt_step_size", 0.3);

    frontier_search_th_ = this->get_parameter("frontier_search_th").as_double();
    int rrt_max_iter = this->get_parameter("rrt_max_iter").as_int();
    rrt_max_r_ = this->get_parameter("rrt_max_r").as_double();
    rrt_min_r_ = this->get_parameter("rrt_min_r").as_double();
    double rrt_step_size = this->get_parameter("rrt_step_size").as_double();

    // Create subscriptions
    RCLCPP_INFO(this->get_logger(), "Subscribe to topics.");
    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        pose_topic, 5, std::bind(&Adsm::pose_callback, this, std::placeholders::_1));
    real_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        real_pose_topic, 5, std::bind(&Adsm::real_pose_callback, this, std::placeholders::_1));
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        laser_topic, 10, std::bind(&Adsm::laser_callback, this, std::placeholders::_1));
    gas_sub_ = this->create_subscription<olfaction_msgs::msg::GasSensor>(
        gas_sensor_topic, 1, std::bind(&Adsm::gas_sensor_callback, this, std::placeholders::_1));
    anemometer_sub_ = this->create_subscription<olfaction_msgs::msg::Anemometer>(
        anemometer_topic, 1, std::bind(&Adsm::anemometer_callback, this, std::placeholders::_1));

    // Create action client
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, nav_action);

    // Create publishers
    visual_points_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("~/visual_points", 1);
    visual_lines_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("~/visual_lines", 1);
    visual_text_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("~/visual_text", 1);
    auto slam_qos = rclcpp::QoS(1).transient_local().reliable();
    slam_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("~/slam_map", slam_qos);

    // Subscribe to external SLAM map (from Python slam_node)
    external_slam_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        external_slam_map_topic, slam_qos,
        std::bind(&Adsm::external_slam_map_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to external SLAM map: %s", external_slam_map_topic.c_str());

    // Initialize algorithm components
    frontier_finder_ = FrontierFinder(frontier_search_th_, obs_r_);
    RCLCPP_INFO(this->get_logger(), "FrontierFinder initialized. find_dis_th %.2f, obs_r %.2f", frontier_search_th_, obs_r_);
    rrt_sampler_ = RRTSampler(rrt_max_iter, rrt_max_r_, obs_r_, rrt_step_size);
    RCLCPP_INFO(this->get_logger(), "RRTSampler initialized. max_iter %d, sample_max_r %.2f, obs_r %.2f, step_size %.2f",
        rrt_max_iter, rrt_max_r_, obs_r_, rrt_step_size);

    RCLCPP_INFO(this->get_logger(), "Adsm node created. Call init() then loop().");
}

void Adsm::init() {
    rclcpp::Rate rate(5);

    // Wait for pose data
    while (rclcpp::ok() && (std::isnan(x_) || std::isnan(real_x_))) {
        RCLCPP_INFO(this->get_logger(), "Waiting for pose topics...");
        rclcpp::spin_some(shared_from_this());
        rate.sleep();
    }

    // Get map from GADEN 3D occupancy service (like efe_igdm)
    {
        std::string service_name = this->get_parameter("gaden_occupancy_service").as_string();
        int z_level = this->get_parameter("z_level").as_int();
        auto client = this->create_client<gaden_msgs::srv::Occupancy>(service_name);
        RCLCPP_INFO(this->get_logger(), "Waiting for GADEN occupancy service '%s'...", service_name.c_str());
        while (rclcpp::ok() && !client->wait_for_service(std::chrono::seconds(2))) {
            RCLCPP_INFO(this->get_logger(), "Still waiting for GADEN occupancy service...");
        }
        auto request = std::make_shared<gaden_msgs::srv::Occupancy::Request>();
        auto future = client->async_send_request(request);
        RCLCPP_INFO(this->get_logger(), "Calling GADEN occupancy service...");
        while (rclcpp::ok() && future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
            rclcpp::spin_some(shared_from_this());
            rate.sleep();
        }
        auto response = future.get();
        int nx = response->num_cells_x;
        int ny = response->num_cells_y;
        int nz = response->num_cells_z;
        double resolution = response->resolution;
        double origin_x = response->origin.x;
        double origin_y = response->origin.y;

        // GADEN origin correction (matches efe_igdm)
        if (origin_x == 0.0 && origin_y == 0.0) {
            origin_x = -0.2;
            origin_y = -0.2;
        }

        if (z_level < 0 || z_level >= nz) {
            RCLCPP_WARN(this->get_logger(), "z_level %d out of bounds [0, %d]. Using z_level=0", z_level, nz - 1);
            z_level = 0;
        }

        // Extract 2D slice and outlet mask at z_level
        // GADEN data layout: index = z * ny * nx + y * nx + x
        const auto& occupancy = response->occupancy;
        std::vector<bool> outlet_mask(nx * ny, false);
        int outlet_count = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx_3d = z_level * ny * nx + y * nx + x;
                if (idx_3d < static_cast<int>(occupancy.size()) && occupancy[idx_3d] == 2) {
                    outlet_mask[y * nx + x] = true;
                    outlet_count++;
                }
            }
        }

        map_.initEmpty(nx, ny, resolution, origin_x, origin_y);
        map_.setOutletMask(outlet_mask);
        RCLCPP_INFO(this->get_logger(), "Map dimensions initialized from GADEN: %dx%d, resolution %.2f, origin (%.2f, %.2f), z_level %d, outlets %d",
            nx, ny, resolution, origin_x, origin_y, z_level, outlet_count);
    }

    // Wait for external SLAM map (from Python slam_node)
    RCLCPP_INFO(this->get_logger(), "Waiting for external SLAM map...");
    while (rclcpp::ok() && !slam_initialized_) {
        rclcpp::spin_some(shared_from_this());
        rate.sleep();
    }
    RCLCPP_INFO(this->get_logger(), "External SLAM map received. Map size: %dx%d", map_.getSizeInCellsX(), map_.getSizeInCellsY());

    // Wait for gas and wind sensors
    while (rclcpp::ok() && (std::isnan(gas_) || std::isnan(wind_speed_))) {
        RCLCPP_INFO(this->get_logger(), "Waiting for gas and anemometer topics...");
        rclcpp::spin_some(shared_from_this());
        rate.sleep();
    }

    // Wait for Nav2 action server
    RCLCPP_INFO(this->get_logger(), "Waiting for Nav2 action server...");
    nav_client_->wait_for_action_server();
    RCLCPP_INFO(this->get_logger(), "Nav2 action server available.");

    // Initialize variables
    iter_ = 1;
    goal_.x = real_x_;
    goal_.y = real_y_;
    last_resample_time_ = this->now().seconds();
    start_rostime_ = this->now().seconds();
    prev_real_x_ = real_x_;
    prev_real_y_ = real_y_;
    total_distance_ = 0.0;

    // Publish SLAM map at 2 Hz for RViz
    slam_map_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&Adsm::publish_slam_map, this));

    RCLCPP_INFO(this->get_logger(), "Adsm initialized.");
}

void Adsm::loop() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    RCLCPP_INFO(this->get_logger(), "Start loop. Rate: %.2f", iter_rate_);
    rclcpp::Rate rate(iter_rate_);
    while (rclcpp::ok()) {
        iter_start_rostime_ = this->now().seconds();
        RCLCPP_INFO(this->get_logger(), "=========================PEAR=========================");
        RCLCPP_INFO(this->get_logger(), "Iteration %2d, ROS time %.2f", iter_, iter_start_rostime_);
        rclcpp::spin_some(shared_from_this());
        observe();
        estimate();
        evaluate();
        navigate();
        record_data();
        if (visual_)
            visualize();
        if (check_terminal()) {
            nav_client_->async_cancel_all_goals();
            save_data();
            rclcpp::shutdown();
        }

        iter_ = iter_ + 1;
        rate.sleep();
    }
}

void Adsm::pose_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    x_ = msg->pose.pose.position.x;
    y_ = msg->pose.pose.position.y;
    z_ = msg->pose.pose.position.z;
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
    tf2::Matrix3x3 m(q);
    m.getRPY(roll_, pitch_, yaw_);
}

void Adsm::real_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    real_x_ = msg->pose.pose.position.x;
    real_y_ = msg->pose.pose.position.y;
    real_z_ = msg->pose.pose.position.z;
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
    tf2::Matrix3x3 m(q);
    m.getRPY(real_roll_, real_pitch_, real_yaw_);
}

void Adsm::laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    // SLAM is now handled by external Python slam_node.
    // This callback is kept as a no-op; the map is updated via external_slam_map_callback.
    (void)msg;
}

void Adsm::external_slam_map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(map_.getMutex());
    map_.update(msg);
    if (!slam_initialized_) {
        slam_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "External SLAM map received: %dx%d, resolution %.2f, origin (%.2f, %.2f)",
            msg->info.width, msg->info.height, msg->info.resolution,
            msg->info.origin.position.x, msg->info.origin.position.y);
    }
}

void Adsm::publish_slam_map() {
    if (!slam_initialized_) return;
    auto grid = map_.getOccupancyGrid();
    grid.header.stamp = this->now();
    slam_map_pub_->publish(grid);
}

void Adsm::gas_sensor_callback(const olfaction_msgs::msg::GasSensor::SharedPtr msg) {
    gas_ = msg->raw;

    double msg_time = rclcpp::Time(msg->header.stamp).seconds();
    gas_msg_queue_.push_back({msg_time, gas_});
    while (msg_time - gas_msg_queue_.front().first > sensor_window_len_) {
        gas_msg_queue_.erase(gas_msg_queue_.begin());
    }

    if (gas_ < gas_low_th_) {
        gas_hit_ = false;
        RCLCPP_INFO(this->get_logger(), "Gas: concentration %.2f < %.2f, hit %d", gas_, gas_low_th_, (int)gas_hit_);
    } else if (gas_ > gas_high_th_) {
        gas_hit_ = true;
        RCLCPP_INFO(this->get_logger(), "Gas: concentration %.2f > %.2f, hit %d", gas_, gas_high_th_, (int)gas_hit_);
    } else {
        bool in_rec = true;
        for (std::size_t i = 1; i < gas_msg_queue_.size(); ++i) {
            if (gas_msg_queue_[i].second - gas_msg_queue_[i-1].second >= 0.0 ||
                gas_msg_queue_[i-1].second - gas_high_th_ >= 0.0) {
                in_rec = false;
            }
        }
        gas_hit_ = in_rec ? false : true;
        RCLCPP_INFO(this->get_logger(), "Gas: concentration %.2f, in_rec %d , hit %d", gas_, (int)in_rec, (int)gas_hit_);
    }
}

void Adsm::anemometer_callback(const olfaction_msgs::msg::Anemometer::SharedPtr msg) {
    wind_speed_ = msg->wind_speed;
    wind_direction_ = msg->wind_direction;
    RCLCPP_INFO(this->get_logger(), "Wind: speed %.2f, direction %.2f", wind_speed_, wind_direction_);
}

double Adsm::probability(double x, double y) {
    double Q = 4.0;
    double D = 1.0;
    double tau = 250;
    double V = wind_speed_;
    double phi = real_yaw_ - wind_direction_;
    double lam = std::sqrt(D * tau / (1 + V * V * tau / (4 * D)));
    double dis = std::sqrt((x - real_x_) * (x - real_x_) + (y - real_y_) * (y - real_y_));
    double dx = real_x_ - x;
    double dy = real_y_ - y;

    double pa = Q / (4 * M_PI * D * (std::abs(dis + 0.0001)));
    double pb = std::exp(-dis / lam);
    double pc = std::exp(-dx * V * std::cos(phi) / (2 * D));
    double pd = std::exp(-dy * V * std::sin(phi) / (2 * D));
    double p = pa * pb * pc * pd;

    return p;
}

bool Adsm::reached_point(double x, double y) {
    for (auto& pose : pose_history_) {
        if (distance(pose.first - x, pose.second - y) < goal_reach_th_) {
            return true;
        }
    }
    return false;
}

void Adsm::create_random_gaol(double start_x, double start_y, double r, double& goal_x, double& goal_y) {
    double rand_theta = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
    double rand_r = static_cast<double>(rand()) / RAND_MAX * r;
    goal_x = start_x + rand_r * cos(rand_theta);
    goal_y = start_y + rand_r * sin(rand_theta);
}

void Adsm::observe() {
    cal_start_time_ = get_current_time();

    RCLCPP_INFO(this->get_logger(), "Pose (gt): x %.2f, y %.2f, yaw %.2f",
        real_x_, real_y_, real_yaw_);

    pose_history_.push_back({real_x_, real_y_});

    do_sample_ = false;
    double dis = distance(real_x_ - goal_.x, real_y_ - goal_.y);
    double t = this->now().seconds();
    double t_duration = t - last_resample_time_;
    if (dis < goal_reach_th_ || t_duration > resample_time_th_) {
        do_sample_ = true;
        last_resample_time_ = t;
    }
    if (do_sample_again_) {
        do_sample_ = true;
        do_sample_again_ = false;
    }
    RCLCPP_INFO(this->get_logger(), "sample: %d, duration %.2f th %.2f, distance %.2f th %.2f",
        (int)do_sample_, t_duration, resample_time_th_, dis, goal_reach_th_);
}

void Adsm::estimate() {
    if (do_sample_) {
        // Resampling using the RRT method
        int MAX_SAMPLE_TIME = 10;
        for (int i = 0; i < MAX_SAMPLE_TIME; ++i) {
            std::vector<RRTNode*> rrt_nodes_temp;
            if (i == 0) {
                rrt_nodes_temp = rrt_sampler_.sample(&map_, real_x_, real_y_);
            } else {
                double new_x, new_y;
                create_random_gaol(real_x_, real_y_, obs_r_, new_x, new_y);
                rrt_nodes_temp = rrt_sampler_.sample(&map_, new_x, new_y);
            }
            RCLCPP_INFO(this->get_logger(), "Sample try: %d/%d, nodes size: %zu", i, MAX_SAMPLE_TIME, rrt_nodes_temp.size());

            if (rrt_nodes_temp.size() < 2) {
                for (RRTNode * node : rrt_nodes_temp) {
                    delete node;
                }
                rrt_nodes_temp.clear();
            } else {
                RCLCPP_INFO(this->get_logger(), "Clear epi_set_ and rrt_nodes_");
                epi_set_.clear();
                for (RRTNode* node : rrt_nodes_) {
                    delete node;
                }
                rrt_nodes_.clear();
                rrt_nodes_ = rrt_nodes_temp;
                break;
            }
        }
    }
    if (rrt_nodes_.size() < 2) {
        do_sample_again_ = true;
    }
    RCLCPP_INFO(this->get_logger(), "do_sample_again_: %d", (int)do_sample_again_);
    if (do_sample_ && (!do_sample_again_)) {
        // Points are divided into goal_cluster_num_ categories according to their angle with the robot
        std::vector<int> categories(rrt_nodes_.size(), -1);
        for (std::size_t i = 0; i < rrt_nodes_.size(); ++i) {
            // Ignore the first element as it is the robot position
            if (i == 0) {
                continue;
            }
            double angle = atan2(rrt_nodes_[i]->y - real_y_, rrt_nodes_[i]->x - real_x_);
            if (angle < 0) {
                angle += 2 * M_PI;
            }
            categories[i] = static_cast<int>((angle / (2 * M_PI)) * goal_cluster_num_);
        }
        std::vector<double> farthest_length(goal_cluster_num_, -1.0);
        std::vector<int> fartest_index(goal_cluster_num_, -1);
        // Find the point in each category that is farthest from the robot
        for (std::size_t i = 0; i < rrt_nodes_.size(); ++i) {
            if (categories[i] < 0) {
                continue;
            }
            double dist = distance(rrt_nodes_[i]->x - real_x_, rrt_nodes_[i]->y - real_y_);
            if (dist > farthest_length[categories[i]]) {
                farthest_length[categories[i]] = dist;
                fartest_index[categories[i]] = i;
            }
        }
        int f_count = 0;
        std::vector<int> is_farthest(rrt_nodes_.size(), 0);
        for (const auto& idx : fartest_index) {
            if (idx >= 0) {
                is_farthest[idx] = 1;
                f_count += 1;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Categories number: %d, max num: %d", f_count, goal_cluster_num_);

        // The farthest point in each category that has frontier points around it will be added to epr_set
        // all other points will be added to epi_set_
        epi_set_.clear();
        for (std::size_t i = 0; i < rrt_nodes_.size(); ++i) {
            if (categories[i] < 0) {
                continue;
            }

            if (!is_farthest[i]) {
                epi_set_.push_back(GoalNode(iter_, rrt_nodes_[i]->x, rrt_nodes_[i]->y, GOAL_EPI_TYPE));
                continue;
            }

            bool do_add_epr = true;
            for (auto& point : epr_set_) {
                double dis = distance(rrt_nodes_[i]->x - point.x, rrt_nodes_[i]->y - point.y);
                if (dis < goal_reach_th_ / 2.0) {
                    do_add_epr = false;
                    break;
                }
            }
            if (!do_add_epr) {
                epi_set_.push_back(GoalNode(iter_, rrt_nodes_[i]->x, rrt_nodes_[i]->y, GOAL_EPI_TYPE));
                continue;
            }

            double f_size = static_cast<double>(frontier_finder_.find(&map_, rrt_nodes_[i]->x, rrt_nodes_[i]->y, false).size());
            if (f_size == 0.0) {
                epi_set_.push_back(GoalNode(iter_, rrt_nodes_[i]->x, rrt_nodes_[i]->y, GOAL_EPI_TYPE));
                continue;
            }

            GoalNode temp_goal_node = GoalNode(iter_, rrt_nodes_[i]->x, rrt_nodes_[i]->y, GOAL_EPR_TYPE);
            temp_goal_node.frontier_size = f_size;
            epr_set_.push_back(temp_goal_node);
        }
        RCLCPP_INFO(this->get_logger(), "size: epi_set %zu, epr_set %zu", epi_set_.size(), epr_set_.size());
    }

    RCLCPP_INFO(this->get_logger(), "Combine epr_set_ and epi_set_.");
    RCLCPP_INFO(this->get_logger(), "Calculate probability and frontier size.");
    goals_.clear();

    std::vector<int> epi_set_delete;
    for (size_t i = 0; i < epi_set_.size(); ++i) {
        if (distance(epi_set_[i].x - real_x_, epi_set_[i].y - real_y_) < rrt_min_r_) {
            epi_set_delete.push_back(i);
        }
    }
    for (int i = epi_set_delete.size() - 1; i >= 0; --i) {
        epi_set_.erase(epi_set_.begin() + epi_set_delete[i]);
    }
    for (auto& temp_goal : epi_set_) {
        temp_goal.probability = gas_hit_ ? probability(temp_goal.x, temp_goal.y) : 0.0;
        temp_goal.frontier_size = 0.0;
        goals_.push_back(temp_goal);
    }

    std::vector<int> epr_set_delete;
    for (size_t i = 0; i < epr_set_.size(); ++i) {
        double dis = distance(real_x_ - epr_set_[i].x, real_y_ - epr_set_[i].y);
        double f_size = epr_set_[i].frontier_size;
        if (dis < rrt_max_r_ + frontier_search_th_) {
            f_size = static_cast<double>(frontier_finder_.find(&map_, epr_set_[i].x, epr_set_[i].y, false).size());
        }
        epr_set_[i].frontier_size = f_size;
        // Remove points whose information gain (number of surrounding frontier points) is 0.
        if (reached_point(epr_set_[i].x, epr_set_[i].y) || f_size == 0.0) {
            epr_set_delete.push_back(i);
        }
    }
    for (int i = epr_set_delete.size() - 1; i >= 0; --i) {
        epr_set_.erase(epr_set_.begin() + epr_set_delete[i]);
    }
    for (auto& temp_goal : epr_set_) {
        temp_goal.probability = gas_hit_ ? probability(temp_goal.x, temp_goal.y) : 0.0;
        temp_goal.frontier_size = temp_goal.frontier_size;
        goals_.push_back(temp_goal);
    }

    RCLCPP_INFO(this->get_logger(), "size: goals_ %zu, epi_set_ %zu, epr_set_ %zu", goals_.size(), epi_set_.size(), epr_set_.size());
}

void Adsm::evaluate() {
    if ((!goals_.empty()) && (!set_random_goal_)) {
        // Normalize j_p, j_i, and j
        RCLCPP_INFO(this->get_logger(), "Calculate j.");
        double sum_probability = 0.0;
        double sum_frontier_size = 0.0;
        for (auto& temp_goal : goals_) {
            sum_probability += temp_goal.probability;
            sum_frontier_size += temp_goal.frontier_size;
        }
        RCLCPP_INFO(this->get_logger(), "sum: probability %.2f, frontier_size %.2f", sum_probability, sum_frontier_size);

        // Static map adaptation: if no gas AND no frontiers, all j=0 so fall back to random exploration
        if (sum_probability == 0.0 && sum_frontier_size == 0.0) {
            RCLCPP_INFO(this->get_logger(), "No gas and no frontiers (static map). Falling back to random exploration.");
            goals_.clear();
            double new_x, new_y;
            create_random_gaol(real_x_, real_y_, random_sample_r_, new_x, new_y);
            goals_.push_back(GoalNode(iter_, new_x, new_y, GOAL_RANDOM_TYPE));
        } else {
            for (auto& temp_goal : goals_) {
                temp_goal.j_p = sum_probability > 0 ? temp_goal.probability / sum_probability : 0.0;
                temp_goal.j_i = sum_frontier_size > 0 ? temp_goal.frontier_size / sum_frontier_size : 0.0;
                double f = temp_goal.iteration / static_cast<double>(iter_);
                temp_goal.j = temp_goal.j_p + k1_ * f * temp_goal.j_i;
            }
        }
    } else {
        goals_.clear();
        RCLCPP_INFO(this->get_logger(), "Generate a random goal. r: %.2f", random_sample_r_);
        double new_x, new_y;
        create_random_gaol(goal_.x, goal_.y, random_sample_r_, new_x, new_y);
        goals_.push_back(GoalNode(iter_, new_x, new_y, GOAL_RANDOM_TYPE));
    }

    // Select the point with the largest j as the navigation goal
    goal_ = goals_[0];
    for (const auto& temp_goal : goals_) {
        if (temp_goal.j > goal_.j) {
            goal_ = temp_goal;
        }
    }

    RCLCPP_INFO(this->get_logger(), "Select navigation goal: x %.2f y %.2f", goal_.x, goal_.y);
    RCLCPP_INFO(this->get_logger(), "goal: j %.2f, j_p %.2f, j_i %.2f, type %d", goal_.j ,goal_.j_p, goal_.j_i, goal_.type);
}

void Adsm::navigate() {
    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose.header.frame_id = map_.getFrameID();
    goal_msg.pose.header.stamp = this->now();
    goal_msg.pose.pose.position.x = goal_.x;
    goal_msg.pose.pose.position.y = goal_.y;
    goal_msg.pose.pose.position.z = 0.0;
    goal_msg.pose.pose.orientation.x = 0.0;
    goal_msg.pose.pose.orientation.y = 0.0;
    goal_msg.pose.pose.orientation.z = 0.0;
    goal_msg.pose.pose.orientation.w = 1.0;

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    nav_client_->async_send_goal(goal_msg, send_goal_options);
    RCLCPP_INFO(this->get_logger(), "Send goal: x %.2f y %.2f", goal_.x, goal_.y);

    cal_duration_ms_ = (get_current_time() - cal_start_time_)*1000;
    RCLCPP_INFO(this->get_logger(), "Loop cost time: %.2f ms", cal_duration_ms_);
}

bool Adsm::check_terminal() {
    // find source check
    dis_to_source_ = distance(real_x_ - source_x_, real_y_ - source_y_);
    if (dis_to_source_ <= source_th_) {
        result_ = "FIND_SOURCE";
        RCLCPP_INFO(this->get_logger(), "FIND_SOURCE: distance %.2f", dis_to_source_);
        return true;
    }

    // max iteration check
    if (iter_ >= max_iter_) {
        result_ = "REACH_MAX_ITER";
        RCLCPP_INFO(this->get_logger(), "REACH_MAX_ITER: iter %d, max_iter %d", iter_, max_iter_);
        return true;
    }

    // robot fall check
    if (abs(real_roll_) > M_PI / 4.0 || abs(real_pitch_) > M_PI / 4.0) {
        result_ = "ROBOT_FALL";
        RCLCPP_INFO(this->get_logger(), "ROBOT_FALL: roll %.2f, pitch %.2f", real_roll_, real_pitch_);
        return true;
    }

    // robot stuck check
    if (std::isnan(stuck_info_[0])) {
        stuck_info_[0] = real_x_;
        stuck_info_[1] = real_y_;
        stuck_info_[2] = this->now().seconds();
    } else {
        double dis = distance(real_x_ - stuck_info_[0], real_y_ - stuck_info_[1]);
        if (dis > 0.15) {
            stuck_info_[0] = real_x_;
            stuck_info_[1] = real_y_;
            stuck_info_[2] = this->now().seconds();
        }
    }

    double stuck_duration = this->now().seconds() - stuck_info_[2];
    if (stuck_duration > stuck_th_) {
        result_ = "ROBOT_STUCK";
        RCLCPP_INFO(this->get_logger(), "ROBOT_STUCK: duration %.2f, th %.2f", stuck_duration, stuck_th_);
        return true;
    }
    set_random_goal_ = stuck_duration > stuck_th_ / 3.0 ? true : false;
    RCLCPP_INFO(this->get_logger(), "dis2source: %.2f th %.2f, stuck_d %.2f th %.2f set_r_goal %d",
        dis_to_source_, source_th_, stuck_duration, stuck_th_, (int)set_random_goal_);

    return false;
}

void Adsm::visualize() {
    RCLCPP_INFO(this->get_logger(), "Visualize.");
    std::string map_frame = map_.getFrameID();

    // targets
    visualization_msgs::msg::Marker marker_t;
    marker_t.ns = "visual_points";
    marker_t.id = 0;
    marker_t.header.frame_id = map_frame;
    marker_t.header.stamp = this->now();
    marker_t.action = visualization_msgs::msg::Marker::ADD;
    marker_t.type = visualization_msgs::msg::Marker::POINTS;
    marker_t.lifetime = rclcpp::Duration::from_seconds(1.0/iter_rate_);
    if (goals_.size() > 1) {
        double max_j = goals_[0].j;
        double min_j = goals_[0].j;
        for (const auto& temp_goal : goals_) {
            if (temp_goal.j > max_j) {
                max_j = temp_goal.j;
            }
            if (temp_goal.j < min_j) {
                min_j = temp_goal.j;
            }
        }
        for (const auto& temp_goal : goals_) {
            geometry_msgs::msg::Point point;
            point.x = temp_goal.x;
            point.y = temp_goal.y;
            point.z = 0.0;
            marker_t.points.push_back(point);
            std_msgs::msg::ColorRGBA goal_color;
            double color_level = max_j > 0 ? temp_goal.j / max_j : 0.0;
            if (temp_goal.type == GOAL_EPI_TYPE) {
                goal_color.r = 1.0 - color_level;
                goal_color.g = 1.0 - color_level;
                goal_color.b = 1.0;
            }
            if (temp_goal.type == GOAL_EPR_TYPE) {
                goal_color.r = 1.0 - color_level;
                goal_color.g = 1.0;
                goal_color.b = 1.0 - color_level;
            }
            goal_color.a = 1.0;
            marker_t.colors.push_back(goal_color);
        }
    }

    // goal
    geometry_msgs::msg::Point point;
    point.x = goal_.x;
    point.y = goal_.y;
    point.z = 0.0;
    marker_t.points.push_back(point);
    std_msgs::msg::ColorRGBA goal_color;
    goal_color.r = 1.0;
    goal_color.g = 0.0;
    goal_color.b = 0.0;
    goal_color.a = 1.0;
    marker_t.colors.push_back(goal_color);

    marker_t.pose.orientation.w = 1.0;
    marker_t.scale.x = 0.05;
    marker_t.scale.y = 0.05;
    marker_t.scale.z = 0.05;
    visual_points_pub_->publish(marker_t);

    // RRT tree
    visualization_msgs::msg::Marker marker_rrt;
    marker_rrt.ns = "visual_lines";
    marker_rrt.id = 0;
    marker_rrt.header.frame_id = map_frame;
    marker_rrt.header.stamp = this->now();
    marker_rrt.action = visualization_msgs::msg::Marker::ADD;
    marker_rrt.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker_rrt.lifetime = rclcpp::Duration::from_seconds(1.0/iter_rate_);
    for (const auto& node : rrt_nodes_) {
        if (node->parent_idx == -1) {
            continue;
        }
        const auto& parent_node = rrt_nodes_[node->parent_idx];
        geometry_msgs::msg::Point parent_point;
        parent_point.x = parent_node->x;
        parent_point.y = parent_node->y;
        parent_point.z = 0.0;
        marker_rrt.points.push_back(parent_point);
        geometry_msgs::msg::Point child_point;
        child_point.x = node->x;
        child_point.y = node->y;
        child_point.z = 0.0;
        marker_rrt.points.push_back(child_point);
    }
    std_msgs::msg::ColorRGBA line_color;
    line_color.r = 0.0;
    line_color.g = 1.0;
    line_color.b = 1.0;
    line_color.a = 1.0;
    marker_rrt.color = line_color;
    marker_rrt.pose.orientation.w = 1.0;
    marker_rrt.scale.x = 0.01;
    marker_rrt.scale.y = 0.01;
    marker_rrt.scale.z = 0.01;
    visual_lines_pub_->publish(marker_rrt);

    // source
    visualization_msgs::msg::Marker marker_source;
    marker_source.ns = "visual_source";
    marker_source.id = 0;
    marker_source.header.frame_id = map_frame;
    marker_source.header.stamp = this->now();
    marker_source.action = visualization_msgs::msg::Marker::ADD;
    marker_source.type = visualization_msgs::msg::Marker::CUBE;
    marker_source.lifetime = rclcpp::Duration::from_seconds(1.0/iter_rate_);
    marker_source.pose.position.x = source_x_;
    marker_source.pose.position.y = source_y_;
    marker_source.pose.position.z = 0.3;
    marker_source.pose.orientation.x = 0.0;
    marker_source.pose.orientation.y = 0.0;
    marker_source.pose.orientation.z = 0.0;
    marker_source.pose.orientation.w = 1.0;
    marker_source.color.r = 0.0;
    marker_source.color.g = 1.0;
    marker_source.color.b = 0.0;
    marker_source.color.a = 1.0;
    marker_source.scale.x = 0.2;
    marker_source.scale.y = 0.2;
    marker_source.scale.z = 0.6;
    visual_lines_pub_->publish(marker_source);

    // text (replaces jsk_rviz_plugins::OverlayText with TEXT_VIEW_FACING marker)
    visualization_msgs::msg::Marker marker_text;
    marker_text.ns = "visual_text";
    marker_text.id = 0;
    marker_text.header.frame_id = map_frame;
    marker_text.header.stamp = this->now();
    marker_text.action = visualization_msgs::msg::Marker::ADD;
    marker_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker_text.lifetime = rclcpp::Duration::from_seconds(1.0/iter_rate_);
    marker_text.pose.position.x = real_x_;
    marker_text.pose.position.y = real_y_;
    marker_text.pose.position.z = 1.0;
    marker_text.pose.orientation.w = 1.0;
    marker_text.scale.z = 0.3;
    marker_text.color.r = 1.0;
    marker_text.color.g = 0.0;
    marker_text.color.b = 0.0;
    marker_text.color.a = 1.0;
    std::string gas_hit_str = gas_hit_ ? "True" : "False";
    std::ostringstream text_ss;
    text_ss << "Iteration: " << iter_ << "\nGas hit: " << gas_hit_str
            << "\nWind direction: " << std::fixed << std::setprecision(2) << wind_direction_;
    marker_text.text = text_ss.str();
    visual_text_pub_->publish(marker_text);
}

void Adsm::record_data() {
    RCLCPP_INFO(this->get_logger(), "Record data.");
    // Accumulate travel distance
    if (!std::isnan(prev_real_x_) && !std::isnan(real_x_)) {
        double step_d = distance(real_x_ - prev_real_x_, real_y_ - prev_real_y_);
        total_distance_ += step_d;
    }
    prev_real_x_ = real_x_;
    prev_real_y_ = real_y_;
    // info
    std::vector<double> info_iter = {
        static_cast<double>(iter_), iter_start_rostime_, cal_duration_ms_, goal_.x, goal_.y, static_cast<double>(goal_.type),
        x_, y_, yaw_, real_x_, real_y_, real_yaw_,
        gas_, gas_hit_, wind_speed_, wind_direction_,
        static_cast<double>(do_sample_), static_cast<double>(epi_set_.size()), static_cast<double>(epr_set_.size())
    };
    info_log_.push_back(info_iter);

    // goals_
    for (auto& target : goals_) {
        std::vector<double> targets_iter = {
            static_cast<double>(iter_), target.iteration, target.x, target.y, static_cast<double>(target.type), target.yaw, target.j,
            target.j_p, target.j_i, target.probability, target.frontier_size
        };
        targets_log_.push_back(targets_iter);
    }

    // rrt_nodes_
    for (RRTNode* node : rrt_nodes_) {
        std::vector<double> rrt_iter = {
            static_cast<double>(iter_), static_cast<double>(node->idx), node->x, node->y, static_cast<double>(node->parent_idx), node->cost
        };
        rrt_log_.push_back(rrt_iter);
    }

    // map_ (info)
    std::vector<double> map_info_iter = {
        double(iter_), map_.getResolution(), static_cast<double>(map_.getSizeInCellsX()), static_cast<double>(map_.getSizeInCellsY()),
        map_.getOriginX(), map_.getOriginY()
    };
    map_info_log_.push_back(map_info_iter);

    // map_ (grid data)
    map_log_.push_back(map_.getData());
}

void Adsm::save_data() {
    double end_time = this->now().seconds();
    double travel_time = end_time - start_rostime_;
    double estimation_error = distance(real_x_ - source_x_, real_y_ - source_y_);
    RCLCPP_INFO(this->get_logger(), "\n"
        "================================================================================\n"
        "ADSM - GAS SOURCE LOCALIZATION - FINAL SUMMARY\n"
        "================================================================================\n"
        "Result:                        %s\n"
        "ST (Search Time):              %d steps\n"
        "TD (Travel Distance):          %.2f m\n"
        "Travel Time:                   %.2f s\n"
        "Estimation Error:              %.3f m\n"
        "Robot Position:                (%.3f, %.3f)\n"
        "True Source:                   (%.3f, %.3f)\n"
        "================================================================================\n",
        result_.c_str(), iter_, total_distance_, travel_time,
        estimation_error, real_x_, real_y_, source_x_, source_y_);
    std::string make_dirs_cmd = "mkdir -p " + data_path_;
    RCLCPP_INFO(this->get_logger(), "%s", make_dirs_cmd.c_str());
    system(make_dirs_cmd.c_str());
    // result.txt
    std::ostringstream result_str;
    result_str << std::fixed << std::setprecision(2)
                << "result:" << result_ << "\n"
                << "x:" << real_x_ << "\n"
                << "y:" << real_y_ << "\n"
                << "roll:" << roll_ << "\n"
                << "pitch:" << pitch_ << "\n"
                << "source_x:" << source_x_ << "\n"
                << "source_y:" << source_y_ << "\n"
                << "distance_to_source:" << dis_to_source_ << "\n"
                << "iteration:" << iter_ << "\n"
                << "max_iteration:" << max_iter_ << "\n"
                << "total_distance:" << total_distance_ << "\n"
                << "travel_time:" << (end_time - start_rostime_) << "\n"
                << "stuck_start_time:" << stuck_info_[2] << "\n"
                << "time:" << end_time << "\n"
                << "run_id:" << random_run_id_;
    std::string result = result_str.str();
    RCLCPP_INFO(this->get_logger(), "%s", result.c_str());
    std::string result_file = data_path_ + "/result.txt";
    RCLCPP_INFO(this->get_logger(), "Save file: %s", result_file.c_str());
    std::ofstream file(result_file);
    if (file.is_open()) {
        file << result;
        file.close();
    } else {
        RCLCPP_ERROR(this->get_logger(), "Unable to open file");
    }

    std::string info_header = "iter,ros_time,cal_duration_ms,goal_x,goal_y,goal_type,";
    info_header += "robot_x,robot_y,robot_yaw,robot_real_x,robot_real_y,robot_real_yaw,";
    info_header += "gas,gas_hit,wind_speed,wind_direction,do_sample,epi_set_size,epr_set_size";
    std::string targets_header = "iteration,iteration_create,x,y,type,yaw,j,j_p,j_i,probability,frontier_size";
    std::string rrt_header = "iter,idx,x,y,parent_idx,cost";
    std::string map_info_header = "iter,resolution,size_x,size_y,origin_x,origin_y";

    save_vector_to_csv(info_log_, data_path_+"/info.csv", info_header);
    save_vector_to_csv(targets_log_, data_path_+"/targets.csv", targets_header);
    save_vector_to_csv(rrt_log_, data_path_+"/rrt_nodes.csv", rrt_header);
    save_vector_to_csv(map_info_log_, data_path_+"/map_info.csv", map_info_header);
    save_gridmap(map_log_, data_path_+"/map.txt");

    // Zip results
    size_t pos = data_path_.find_last_of('/');
    std::string dir_name = data_path_.substr(pos + 1);
    std::string zip_cmd = "cd " + data_path_ + "/.. && zip -rm " + dir_name + ".zip " + dir_name + " > /dev/null 2>&1";
    RCLCPP_INFO(this->get_logger(), "%s", zip_cmd.c_str());
    system(zip_cmd.c_str());
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Adsm>();
    node->init();
    node->loop();
    rclcpp::shutdown();
    return 0;
}

void save_vector_to_csv(const std::vector<std::vector<double>> &data, const std::string &filename, const std::string &header) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    if (!header.empty()) {
        file << header << "\n";
    }

    file << std::fixed << std::setprecision(4);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

void save_gridmap(std::vector<std::vector<int8_t>> data, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << (int)row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

Adsm::~Adsm() {
    for (RRTNode* node : rrt_nodes_) {
        delete node;
    }
    rrt_nodes_.clear();
}
