# OSL on GADEN

This repository will contain our ROS2 packages.

**Repository Structure:**
```
~/<your-ros2-workspace>         # e.g., gaden_ros_files (for netlab user)
└── src
    └── base                    # This repository
        ├── README.md
        └── <your-package>      # e.g., infotaxis
```

To run the packages: 
```bash
cd ~/<your-ros2-workspace>
colcon build
```