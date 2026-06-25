import time
from functools import cached_property
from functools import wraps
from threading import Thread
from typing import Optional




from lerobot.cameras.camera import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError
from lerobot.utils.errors import DeviceNotConnectedError

from .config import ActionType
from .config import ObservationType
from .config import ROS2RobotConfig

from pymoveit2 import MoveIt2
from pymoveit2 import GripperInterface
from pymoveit2 import MoveIt2Servo

import rclpy
from rclpy.node import Node
from rclpy.node import Client
from rclpy.node import Publisher
from rclpy.node import Subscription
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

from sensor_msgs.msg import JointState

from std_msgs.msg import Header

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist


def executor_safe(f):
    import rclpy
    from functools import wraps
    from rclpy.node import Node
    @wraps(f)
    def decorator(*args, **kwargs):
        node = args[0].node
        if not isinstance(node, Node):
            raise RuntimeError(f"{f.__name__} is not tied to a ROS 2 Node.")
            
        mte = node.executor
        
        # Cleanly detach from current executor
        if mte is not None:
            mte.remove_node(node)
            node.executor = None

        try:
            result = f(*args, **kwargs)
        finally:
            # Cleanly detach from global executor used by spin_until_future_complete
            ge = rclpy.get_global_executor()
            try:
                ge.remove_node(node)
            except BaseException:
                pass
            node.executor = None
            
            # Restore to original executor
            if mte is not None:
                mte.add_node(node)
                mte.wake()
                
        return result
    return decorator

class ROS2Robot(Robot):
    def __init__(self, config: ROS2RobotConfig):
        self.name = config.id if config.id is not None else f"ROS2Robot_{config.namespace.replace('/', '_')}"
        
        super().__init__(config)
        
        self.config = config

        self.node: Optional[Node] = None
        self._reentrant_callback_group: Optional[ReentrantCallbackGroup] = None
        
        self._moveit2: Optional[MoveIt2] = None
        self._gripper_interface: Optional[GripperInterface] = None
        self._servo_interface: Optional[MoveIt2Servo] = None
        
        self.joint_state_sub: Subscription = None
        self.joint_state: JointState = None
        
        self.joint_command_sub: Subscription = None
        self.joint_command: JointState = None
        
        self.joint_command_pub: Publisher = None
        self.directly_publish: bool = config.directly_publish
        
        self.start = None
        
        # Initialize cameras
        self.cameras: dict[str, Camera] = make_cameras_from_configs(self.config.cameras)
        
    def __del__(self):
        if self.is_connected:
            self.disconnect()    
    
    def _init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        if self.node is None:
            self.node = Node(
                node_name=self.id, 
                namespace=self.config.namespace,
                parameter_overrides=[
                    rclpy.Parameter(name='use_sim_time', value=True)
                ]
            )
            self._reentrant_callback_group = ReentrantCallbackGroup()

    def _ros_thread(self):
        if self.node is not None:
            self.executor: MultiThreadedExecutor = MultiThreadedExecutor()
            self.executor.add_node(self.node)

            self.executor.spin_until_future_complete(self.start)
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.node = None
            
            self.executor.shutdown()
        
    def connect(self):
    
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Robot is already connected.")
    
        # ROS2 initialization
        self._init_ros()
          
        # Connect to MoveIt2 and Gripper interfaces based on the specified action types
        if (self.config.arm_action_type == ActionType.JOINT_POSITION 
            or self.config.arm_action_type == ActionType.JOINT_TRAJECTORY
            or self.config.arm_action_type == ActionType.CARTESIAN_POSE):
            self.init_moveit2()
        
        if (self.config.gripper_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY]):
            self.init_gripper_interface()
            
        if (self.config.arm_action_type == ActionType.CARTESIAN_VELOCITY):
            self.init_servo_interface()
        
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            'joint_states',
            self._jointStateCallback,
            1,
            callback_group=self._reentrant_callback_group
        )
        
        self.joint_command_sub = self.node.create_subscription(
            JointState,
            'joint_command',
            self._jointCommandCallback,
            1,
            callback_group=self._reentrant_callback_group
        )
        
        self.joint_command_pub = self.node.create_publisher(
            JointState,
            'joint_command',
            1,
            callback_group=self._reentrant_callback_group
        )
        
        for cam in self.cameras.values():
            cam.connect()

        if self.start is None:
            self.start = Future()
            self.ros_thread = Thread(target=self._ros_thread)
            self.ros_thread.start()

        self._wait_for_joint_state()

    def _wait_for_joint_state(self, timeout: float = None) -> bool:
        rate = self.node.create_rate(0.01, self.node.get_clock())
        count = 0
        while self.joint_state is None and rclpy.ok() and self.node is not None:
            if count%100 == 1:
                self.node.get_logger().warn(f'Waiting for {self.config.namespace} for first message.') 
            time.sleep(0.01)
            count += 1
            if timeout is not None and timeout >= count*10:
                break
        # Cleanup
        rate.destroy()
        
        return self.joint_state is not None
        
        
    def _jointStateCallback(self, msg: JointState):
        if any(name in msg.name for name in self.config.arm_joint_names + self.config.gripper_joint_names):
            self.joint_state = msg
            
    def _jointCommandCallback(self, msg: JointState):
        if any(name in msg.name for name in self.config.arm_joint_names + self.config.gripper_joint_names):
            self.joint_command = msg
        
    @property
    def is_connected(self) -> bool:
        return self.node is not None and all(cam.is_connected for cam in self.cameras.values())
        
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_obs_ft, **self._cameras_ft}
        
    @cached_property
    def action_features(self) -> dict[str, type | tuple]:
        return {**self._motors_act_ft}
        
    @property
    def _motors_obs_ft(self) -> dict[str, type]:
        if self.config.arm_observation_type == ObservationType.JOINT_POSITION:
            arm_ft = {f'{self.config.arm_joint_names[i]}.pos': float for i in range(len(self.config.arm_joint_names))}
        if self.config.arm_observation_type == ObservationType.JOINT_VELOCITY:
            arm_ft = {f'{self.config.arm_joint_names[i]}.vel': float for i in range(len(self.config.arm_joint_names))}
        if self.config.arm_observation_type == ObservationType.JOINT_EFFORT:
            arm_ft = {f'{self.config.arm_joint_names[i]}.effort': float for i in range(len(self.config.arm_joint_names))}
        if self.config.gripper_observation_type == ObservationType.JOINT_POSITION:
            gripper_ft = {f'{self.config.gripper_joint_names[i]}.pos': float for i in range(len(self.config.gripper_joint_names))}
        if self.config.gripper_observation_type == ObservationType.JOINT_VELOCITY:
            gripper_ft = {f'{self.config.gripper_joint_names[i]}.vel': float for i in range(len(self.config.gripper_joint_names))}
        if self.config.gripper_observation_type == ObservationType.JOINT_EFFORT:
            gripper_ft = {f'{self.config.gripper_joint_names[i]}.effort': float for i in range(len(self.config.gripper_joint_names))}
        return {**arm_ft}
        return {**arm_ft, **gripper_ft}
        
    @property
    def _motors_act_ft(self) -> dict[str, type]:
        if self.config.arm_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY]:
            arm_ft = {f'{self.config.arm_joint_names[i]}.pos': float for i in range(len(self.config.arm_joint_names))}
        if self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            arm_ft = {
                "x": float,
                "y": float,
                "z": float,
                "qx": float,
                "qy": float,
                "qz": float,
                "qw": float
            }
        if self.config.arm_action_type == ActionType.CARTESIAN_VELOCITY:
            arm_ft = {
                "x": float,
                "y": float,
                "z": float,
                "wx": float,
                "wy": float,
                "wz": float
            }
        if self.config.gripper_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY]:
            gripper_ft = {f'{self.config.gripper_joint_names[i]}.pos': float for i in range(len(self.config.gripper_joint_names))}
        return {**arm_ft, **gripper_ft}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    def init_moveit2(self) -> MoveIt2:
        if self._moveit2 is not None:
            raise DeviceAlreadyConnectedError("MoveIt2 interface is already connected.")
        
        self._moveit2 = MoveIt2(
            node=self.node,
            joint_names=self.config.arm_joint_names,
            base_link_name=self.config.base_link_name,
            end_effector_name=self.config.end_effector_name,
            group_name=self.config.group_name,
            callback_group=self._reentrant_callback_group,
        )
        self._moveit2.planner_id = self.config.planner_id
        
        self._moveit2.max_velocity = self.config.max_velocity
        self._moveit2.max_acceleration = self.config.max_acceleration
        
        return self._moveit2

    def init_gripper_interface(self) -> GripperInterface:
        if self._gripper_interface is not None:
            raise DeviceAlreadyConnectedError("Gripper command interface is already connected.")
        
        self._gripper_interface = GripperInterface(
            node=self.node,
            gripper_joint_names=self.config.gripper_joint_names,
            open_gripper_joint_positions=self.config.open_gripper_joint_positions,
            closed_gripper_joint_positions=self.config.closed_gripper_joint_positions,
            gripper_group_name=self.config.gripper_group_name,
            max_effort=self.config.gripper_max_effort,
            callback_group=self._reentrant_callback_group,
            gripper_command_action_name=self.config.gripper_command_action_name,
        )
        
        return self._gripper_interface
        
    def init_servo_interface(self) -> MoveIt2Servo:
        if self._servo_interface is not None:
            raise DeviceAlreadyConnectedError("Servo interface is already connected.")
        
        self._servo_interface = MoveIt2Servo(
            node=self.node,
            frame_id=self.config.servo_frame_id,
            namespace=self.config.namespace,
            callback_group=self._reentrant_callback_group,
        )
        
        return self._servo_interface
        
    @property
    def is_calibrated(self):
        return True
    
    def calibrate(self):
        pass
        
    def configure(self):
        pass
        
    # TODO: Add support for cartesian position, cartesian velocity observations
    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is not connected.")
        
        obs_dict = {}
        for joint_name in self.config.arm_joint_names:
            if joint_name in self.joint_state.name:
                idx = self.joint_state.name.index(joint_name)
                if self.config.arm_observation_type == ObservationType.JOINT_POSITION:
                    obs_dict[f'{joint_name}.pos'] = self.joint_state.position[idx]
                if self.config.arm_observation_type == ObservationType.JOINT_VELOCITY:
                    obs_dict[f'{joint_name}.vel'] = self.joint_state.velocity[idx]
                if self.config.arm_observation_type == ObservationType.JOINT_EFFORT:
                    obs_dict[f'{joint_name}.eff'] = self.joint_state.effort[idx]
                    
        for joint_name in self.config.gripper_joint_names:
            if joint_name in self.joint_state.name:
                idx = self.joint_state.name.index(joint_name)
                if self.config.gripper_observation_type == ObservationType.JOINT_POSITION:
                    obs_dict[f'{joint_name}.pos'] = self.joint_state.position[idx]
                if self.config.gripper_observation_type == ObservationType.JOINT_VELOCITY:
                    obs_dict[f'{joint_name}.vel'] = self.joint_state.velocity[idx]
                if self.config.gripper_observation_type == ObservationType.JOINT_EFFORT:
                    obs_dict[f'{joint_name}.eff'] = self.joint_state.effort[idx]
                    
        for cam_name, cam in self.cameras.items():
            obs_dict[cam_name] = cam.read()
            
        return obs_dict
        
    def get_action(self) -> dict[str, float]:
        if self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            arm_action = {
                f'{joint_name}.pos': 
                self.joint_command.position[self.joint_command.name.index(joint_name)] 
                for joint_name in self.config.arm_joint_names
            }
                
            return arm_action
            
        raise NotImplementedError("get_action is only implemented for CARTESIAN_POSE action type for now, and will need to be implemented for other action types in the future.")
        
            

    def send_action(self, action: Pose | Twist | dict[str, float], cartesian=False, wait_for_execution=False) -> bool:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is not connected.")
            
        action_executed = False
        
        # 1. Arm Action Execution
        if self.config.arm_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY]:
            if isinstance(action, dict):
                arm_goal_pos = {k.removesuffix('.pos'): v for k, v in action.items() if k.endswith('.pos') and k.removesuffix('.pos') in self.config.arm_joint_names}
                
                if self.directly_publish and arm_goal_pos:
                    joint_command_msg = JointState()
                    joint_command_msg.header = Header()
                    joint_command_msg.header.stamp = self.node.get_clock().now().to_msg()
                    joint_command_msg.header.frame_id = self.config.frame_id
                    joint_command_msg.name = list(arm_goal_pos.keys())
                    joint_command_msg.position = list(arm_goal_pos.values())
                    self.joint_command_pub.publish(joint_command_msg)
                    action_executed = True
                
                elif self.config.arm_action_type == ActionType.JOINT_POSITION and arm_goal_pos:
                    if all(joint in arm_goal_pos for joint in self.config.arm_joint_names):
                        success = self._moveit2.move_to_configuration([arm_goal_pos[joint] for joint in self.config.arm_joint_names])
                        if not success:
                            self.node.get_logger().error("Failed to move to joint configuration.")
                            if self.config.fallback_planner_id:
                                self.node.get_logger().info("Trying fallback planner.")
                                self._moveit2.planner_id = self.config.fallback_planner_id
                                success = self._moveit2.move_to_configuration([arm_goal_pos[joint] for joint in self.config.arm_joint_names])
                                if not success:
                                    self.node.get_logger().error("Failed to move to joint configuration with fallback planner.")
                                else:
                                    self.node.get_logger().info("Successfully moved to joint configuration with fallback planner.")
                                    action_executed = True
                                self._moveit2.planner_id = self.config.planner_id
                        else:
                            action_executed = True
                            
                        if action_executed and wait_for_execution:
                            self._moveit2.wait_until_executed()
                    else:
                        self.node.get_logger().warn("Incomplete arm joint configuration provided in action.")
                
                elif self.config.arm_action_type == ActionType.JOINT_TRAJECTORY and arm_goal_pos:
                    self.node.get_logger().warn("JOINT_TRAJECTORY action type is not yet implemented.")
                    
        elif self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            if isinstance(action, Pose):
                pose_stamped = PoseStamped()
                pose_stamped.header = Header()
                pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
                pose_stamped.header.frame_id = self.config.frame_id
                pose_stamped.pose = action
                success = self._moveit2.move_to_pose(
                    pose=pose_stamped,
                    cartesian=cartesian,
                    start_joint_state=self.joint_state
                )
                if not success:
                    self.node.get_logger().error("Failed to move to Cartesian pose.")
                    if self.config.fallback_planner_id:
                        self.node.get_logger().info("Trying fallback planner.")
                        self._moveit2.planner_id = self.config.fallback_planner_id
                        success = self._moveit2.move_to_pose(
                            pose=pose_stamped,
                            cartesian=cartesian,
                            start_joint_state=self.joint_state
                        )
                        if not success:
                            self.node.get_logger().error("Failed to move to Cartesian pose with fallback planner.")
                        else:
                            self.node.get_logger().info("Successfully moved to Cartesian pose with fallback planner.")
                            action_executed = True
                        self._moveit2.planner_id = self.config.planner_id
                else:
                    action_executed = True
                    
                if action_executed and wait_for_execution:
                    self._moveit2.wait_until_executed()
            elif isinstance(action, dict):
                arm_keys = [k for k in action.keys() if k.removesuffix('.pos') in self.config.arm_joint_names]
                if arm_keys:
                    self.node.get_logger().error("Invalid action type for CARTESIAN_POSE arm action. Expected Pose object.")
                    
        elif self.config.arm_action_type == ActionType.CARTESIAN_VELOCITY:
            if isinstance(action, Twist):
                self._servo_interface.servo(
                    linear=[action.linear.x, action.linear.y, action.linear.z],
                    angular=[action.angular.x, action.angular.y, action.angular.z]
                )
                action_executed = True
            elif isinstance(action, dict):
                arm_keys = [k for k in action.keys() if k.removesuffix('.pos') in self.config.arm_joint_names]
                if arm_keys:
                    self.node.get_logger().error("Invalid action type for CARTESIAN_VELOCITY arm action. Expected Twist object.")

        # 2. Gripper Action Execution
        if self.config.gripper_action_type == ActionType.JOINT_POSITION:
            if isinstance(action, dict):
                gripper_goal_pos = {
                    k.removesuffix('.pos'): v 
                    for k, v in action.items() 
                    if k.endswith('.pos') and k.removesuffix('.pos') in self.config.gripper_joint_names
                }
                gripper_goal_vals = list(gripper_goal_pos.values())
                if gripper_goal_vals:
                    if len(gripper_goal_vals) == 1:
                        self._gripper_interface.move_to_position(gripper_goal_vals[0])
                        action_executed = True
                        if wait_for_execution:
                            self._gripper_interface.wait_until_executed()
                    else:
                        self.node.get_logger().warn(f"Received gripper action with {len(gripper_goal_vals)} joints, but only single-joint control is currently supported.")
        
        return action_executed

    def disconnect(self):
        for cam in self.cameras.values():
            print(f'Disconnecting camera: {cam.topic}')
            cam.disconnect()
        
        print('Shutting down ROS node...')
        self.start.set_result(True)
        self.ros_thread.join()
        
        self.node = None
        self.start = None
        
        self.joint_state = None
        self.joint_command = None
        
        self._moveit2 = None
        self._gripper_interface = None
        self._servo_interface = None

    def sleep(self, seconds: float):
        rate = self.node.create_timer(seconds)
        rate.sleep()
        rate.destroy()

    @executor_safe
    def callService(self, service: Client, request, message: str | None = None):
        if isinstance(message, str):
            self.node.get_logger().info(message)
        self.node.get_logger().info(f"Calling {service.srv_name} service.")
        
        response = service.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        response = response.result()
        
        self.node.get_logger().info(f"Called {service.srv_name} successfully.")
        return response