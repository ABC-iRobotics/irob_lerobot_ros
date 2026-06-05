# LeRobot ROS 2 Integration

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Repo size](https://img.shields.io/github/repo-size/ABC-iRobotics/lerobot_ros)
![Repo stars](https://img.shields.io/github/stars/ABC-iRobotics/lerobot_ros)
![Repo forks](https://img.shields.io/github/forks/ABC-iRobotics/lerobot_ros)

## Introduction
The `lerobot_ros` package provides a seamless ROS 2 integration for Hugging Face's LeRobot framework. It acts as a bridge, allowing users to control ROS 2-enabled robotic arms and cameras directly through the standard LeRobot interface.

The package implements the `ROS2Robot` and `ROS2Camera` classes, which wrap standard ROS 2 publishers, subscribers, MoveIt 2, and MoveIt Servo interfaces to be fully compatible with LeRobot's dataset collection and policy execution pipelines.

### Inspiration & Key Contributions
This project was heavily inspired by the foundational work in [ycheng517/lerobot-ros](https://github.com/ycheng517/lerobot-ros). Building upon that concept, this repository introduces several significant contributions and architectural improvements:

- **MoveIt 2 Integration:** Instead of relying solely on direct controller publishing (`directly_publish`), this package natively integrates MoveIt 2. This enables robust motion planning, collision avoidance, and trajectory execution in both joint space and Cartesian space.
- **MoveIt Servo Support:** Added support for real-time Cartesian velocity control using MoveIt Servo, allowing for smooth and responsive teleoperation or policy execution.
- **Extended Action Spaces:** Full compatibility with various action and observation formulations, including Cartesian poses, joint positions, and velocities.
- **Real-Time Vision Bridging:** Integrated `cv_bridge` to seamlessly process real-time ROS 2 Image topics into LeRobot-compatible camera observation formats.

## Prerequisites

- ROS 2 Humble
- MoveIt 2 and MoveIt Servo
- OpenCV and cv_bridge
- Hugging Face LeRobot

## Setup guide

Using the previous section's links, install every prerequisites.

Navigate into your ROS 2 workspace and copy this repo to your source folder:
```bash
cd src && git clone https://github.com/ABC-iRobotics/lerobot_ros.git && cd ..
```

Build the package:
```bash
colcon build --packages-select lerobot_ros
```

## Usage

After installation, you can initialize a `ROS2Robot` object in your Python scripts using the `ROS2RobotConfig`.

```python
from lerobot_ros.config import FR3RobotConfig, ROS2CameraConfig, ActionType
from lerobot_ros.ros2robot import ROS2Robot

# Configure the robot
config = FR3RobotConfig(
    frame_id="world",
    namespace="/franka",
    planner_id="LazyPRMstarkConfigDefault"
)
config.cameras = {
    'base': ROS2CameraConfig(namespace="", frame_id="base", topic="base_camera/image_raw")
}
config.arm_action_type = ActionType.CARTESIAN_POSE

# Initialize and connect
robot = ROS2Robot(config=config)
robot.connect()

# Read observation
obs = robot.get_observation()

# Send action
robot.send_action(action_msg)

# Disconnect
robot.disconnect()
```

## Troubleshooting

In case of any issues, check the official resources:
- Hugging Face LeRobot Documentation
- ROS 2 Humble Documentation
- MoveIt 2 Tutorials

## Contributing

Contributions to the `lerobot_ros` package are highly appreciated! If you would like to help improve the package or integrate new features, please take a look at our [TODO.md](./TODO.md) file, which lists the currently known issues and areas needing development.

## Author

András Makány - Graduate student at Obuda University

## License

This software is released under the GPL-3.0-only License, see package.xml for details.
