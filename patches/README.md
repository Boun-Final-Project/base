# Patches for third-party dependencies

These patches must be applied to external repos after cloning them.

---

## test_env_launch.patch

**Target:** `src/gaden/test_env/launch/main_simbot_launch.py`

**What it adds:**
- `robot_x` / `robot_y` launch arguments to override robot start position at launch time (writes a temp BasicSimScene.yaml)
- `speed` launch argument (was hardcoded to 1.0)
- `method` launch argument to select the GSL agent (efe_igdm, efe_igdm_wind, igdm_multiple)
- `playback` and `initial_iteration` arguments for gaden_player
- `use_rviz` argument

**How to apply:**

```bash
cd ~/ros2_ws/src/gaden
git apply ~/ros2_ws/src/base/patches/test_env_launch.patch
```

Then rebuild:

```bash
cd ~/ros2_ws
colcon build --packages-select test_env --symlink-install
source install/setup.bash
```

**To verify the patch applies cleanly** (dry run):

```bash
cd ~/ros2_ws/src/gaden
git apply --check ~/ros2_ws/src/base/patches/test_env_launch.patch
```

> If the patch fails (e.g. upstream changed the file), resolve conflicts manually
> by applying the changes described above to `test_env/launch/main_simbot_launch.py`.
