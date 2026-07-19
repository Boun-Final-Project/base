# ADSM — ROS 2 Humble Port

ROS 2 Humble port of the [An Adaptive Robot Search Algorithm for Balancing Exploitation and Exploration in Indoor Intermittent Source Seeking](https://ieeexplore.ieee.org/document/11297772/) by Wang et al. (IEEE TIE, 2025).

- **Original ROS 1 Noetic implementation:** [mwanggh/An-adaptive-robot-search-algorithm](https://github.com/mwanggh/An-adaptive-robot-search-algorithm)
- **DOI:** [10.1109/TIE.2025.3632565](https://doi.org/10.1109/TIE.2025.3632565)

## Changes from the original (ROS 1 → ROS 2)

| Component | Original (ROS 1 Noetic) | This port (ROS 2 Humble) |
|---|---|---|
| Build system | catkin / CMake | ament_cmake |
| ROS node API | `ros::NodeHandle` / `ros::Rate` | `rclcpp::Node` / `rclcpp::Rate` |
| Logging | `ROS_INFO` / `ROS_ERROR` | `RCLCPP_INFO` / `RCLCPP_ERROR` |
| Gas sensor | MOX TGS2620 (`sensor_model=0`, resistance) | PID MiniRaeLite (`sensor_model=30`, PPM) |
| Navigation | `move_base` action (ROS 1) | Nav2 `navigate_to_pose` action (ROS 2) |
| Pose source | SLAM odom (`/odom`) | Ground truth (`/ground_truth`) |
| Map source | `/map` from gmapping | Custom Python SLAM + GADEN occupancy service |
| Launch | XML `.launch` | Python `LaunchDescription` |
| UUID generation | Boost UUID | Custom `std::random_device` generator |
| Added features | — | Outlet mask from GADEN 3D occupancy (unused, no algorithmic effect) |

## Equivalence test

A standalone C++ test compares the port's decision math **verbatim** against the original source across 2.3 million randomized inputs:

| Component | Test cases | Mismatches | Result |
|---|---|---|---|
| `probability()` — Gaussian plume model (Eq. 3) | 2,000,000 | **0** | Bit-exact |
| `evaluate()` — fitness + argmax goal selection | 199,769 | **0** | Bit-exact |
| Angular clustering — N-class bearing split, farthest-per-class | 100,000 | **0** | Bit-exact |
| Gas binarization — hysteresis thresholding | 200,000 | differs | Documented: MOX → PID sensor conversion |

**Verdict:** The decision core is **bit-exact equivalent** to the original authors' code. The only divergence is the intentional, documented MOX-to-PID gas sensor conversion.

Four shared algorithm files (`frontier_finder.cpp`, `rrt_sampler.cpp`, `goal.cpp`, and the algorithmic body of `adsm.cpp`) differ only in include paths, logging macros, and whitespace — no semantic changes.

### Running the equivalence test

```bash
cd /tmp/adsm_equiv
g++ -O2 -o equiv_test equiv_test.cpp -lm
./equiv_test
```

## Parameters

All algorithm parameters are identical to the original paper (Section IV-C):

```
k1=0.2, random_sample_r=3.0, goal_cluster_num=20, obs_r=0.2
goal_reach_th=0.5, resample_time_th=5.5, sensor_window_length=6.0
rrt_max_iter=200, rrt_max_r=3.0, rrt_min_r=0.70, rrt_step_size=0.3
frontier_search_th=3.0, stuck_duration_th=60.0, iter_rate=1.0, max_iter=360
```

Gas thresholds are re-scaled for the PID sensor:

```
gas_max=10.0, gas_high_th=0.3, gas_low_th=0.1
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select adsm
```

## Run

```bash
ros2 launch adsm adsm_launch.py
```

Or via the benchmark driver:

```bash
cd ~/ros2_ws
./run_adsm_eesa_benchmark.sh
```

## Dependencies

- ROS 2 Humble
- `rclcpp`, `tf2`, `tf2_geometry_msgs`, `nav_msgs`, `geometry_msgs`
- `olfaction_msgs` (GADEN sensor messages)
- `Nav2` (`navigate_to_pose` action server)
- GADEN player + simulated gas sensor + simulated anemometer

## License

This is a scientific reimplementation of a published algorithm. The original authors' work is cited above; the algorithm specification is publicly available in the IEEE paper. This port is provided for academic comparison and reproducibility.

## Citation

```bibtex
@ARTICLE{wang2025adaptive,
  author={Wang, Miao and Xin, Bin and Deng, Fang and Chen, Chen and Qu, Yun},
  journal={IEEE Transactions on Industrial Electronics},
  title={An Adaptive Robot Search Algorithm for Balancing Exploitation
         and Exploration in Indoor Intermittent Source Seeking},
  year={2025},
  doi={10.1109/TIE.2025.3632565}
}
```
