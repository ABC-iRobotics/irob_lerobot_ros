# To-Do List

This document keeps track of the currently known issues and planned features for the `irob_lerobot_ros` package.

## `irob_lerobot_ros/ros2robot.py`
- **Servo command error handling:** Add error handling for servo command execution and return `False` if execution fails (in `send_action()`).
- **Gripper action error handling:** Add error handling for gripper action execution and return `False` if execution fails (in `send_action()`).
- **Cartesian observation type:** Add support for cartesian position and velocity observations (in `get_observation()`).
- **Joint trajectory action type:** Add support for joint trajectory action type (in `send_action()`).

## `irob_lerobot_ros/ros2camera.py`
- **Camera info waiting:** Implement waiting for camera info and distortion parameters (in `calibrate()`).
- **Calibration logic:** Implement logic for when the camera is calibrated (in `is_calibrated()`).
- **Distortion correction:** Implement camera correction for distortion on the incoming image stream (in `_cameraImageCallback()`).
