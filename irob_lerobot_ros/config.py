from dataclasses import dataclass, field
from typing import List

from enum import Enum

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig

class ActionType(Enum):
    CARTESIAN_POSE = "cartesian_pose"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    JOINT_POSITION = "joint_position"
    JOINT_TRAJECTORY = "joint_trajectory"
    
class ObservationType(Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_EFFORT = "joint_effort"

@dataclass
class ROS2CameraConfig(CameraConfig):
    type: str = "ROS2Camera"
    namespace: str = ""
    topic: str = "/camera/image_raw"
    frame_id: str = "camera_link"

@dataclass
class ROS2RobotConfig(RobotConfig):
    namespace: str = "/robot"
    arm_action_type: ActionType = ActionType.JOINT_POSITION
    gripper_action_type: ActionType = ActionType.JOINT_POSITION
    
    arm_observation_type: ObservationType = ObservationType.JOINT_POSITION
    gripper_observation_type: ObservationType = ObservationType.JOINT_POSITION
    
    frame_id: str = "base_link"
    
    # MoveIt2 specific configurations
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "joint1", 
            "joint2", 
            "joint3", 
            "joint4", 
            "joint5", 
            "joint6"
        ]
    )
    base_link_name: str = "base_link"
    end_effector_name: str = "ee_link"
    group_name: str = "arm"
    pipeline_id: str = "ompl"
    planner_id: str = "RRTConnectkConfigDefault"
    fallback_planner_id: str = "BiESTkConfigDefault"
    
    min_joint_positions: List[float] = field(
        default_factory=lambda: [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
    )
    max_joint_positions: List[float] = field(
        default_factory=lambda: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
    )
    
    max_velocity: float = 1.0
    max_acceleration: float = 1.0
    
    ##########################################################################
    
    # GripperInterface/MoveIt2Gripper specific configurations
    gripper_joint_names: List[str] = field(
        default_factory=lambda: [
            "gripper_joint"
        ]
    )
    open_gripper_joint_positions: List[float] = field(
        default_factory=lambda: [0.0]
    )
    closed_gripper_joint_positions: List[float] = field(
        default_factory=lambda: [0.8]
    )
    gripper_max_effort: float = 10.0
    gripper_group_name: str = "gripper"
    
    gripper_command_action_name: str = f"{gripper_group_name}_controller/gripper_cmd"
    
    ##########################################################################
    
    # MoveIt2Servo specific configurations
    servo_frame_id: str = end_effector_name
    servo_linear_speed: float = 1.0
    servo_angular_speed: float = 1.0
    
    
# * Example configurations
    
@RobotConfig.register_subclass("fr3")
@dataclass
class FR3RobotConfig(ROS2RobotConfig):
    id: str = "franka"
    namespace: str = "/franka"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "fr3_joint1", 
            "fr3_joint2", 
            "fr3_joint3", 
            "fr3_joint4", 
            "fr3_joint5", 
            "fr3_joint6",
            "fr3_joint7"
        ]
    )
    frame_id: str = "fr3_link0"
    base_link_name: str = "fr3_link0"
    end_effector_name: str = "fr3_hand_tcp"
    group_name: str = "fr3_arm"
    pipeline_id: str = "ompl"
    planner_id: str = "RRTConnectkConfigDefault"

    directly_publish: bool = False
    
    max_joint_positions: List[float] = field(
        default_factory=lambda: [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]
    )
    min_joint_positions: List[float] = field(
        default_factory=lambda: [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
    )

    max_velocity: float = 0.5
    max_acceleration: float = 0.5
    
    gripper_joint_names: List[str] = field(
        default_factory=lambda: [
            "fr3_finger_joint1"
        ]
    )
    open_gripper_joint_positions: List[float] = field(
        default_factory=lambda: [0.0]
    )
    closed_gripper_joint_positions: List[float] = field(
        default_factory=lambda: [0.04]
    )
    
    gripper_group_name: str = "franka_gripper"
    gripper_command_action_name: str = "franka_gripper/gripper_cmd"
    