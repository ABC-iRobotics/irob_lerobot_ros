import time
from collections import deque
from functools import cached_property
from functools import wraps
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation




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
from pymoveit2.utils import spin_or_wait

import rclpy
from rclpy.node import Node
from rclpy.node import Client
from rclpy.node import Publisher
from rclpy.node import Subscription
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from rclpy.duration import Duration

from sensor_msgs.msg import JointState

from std_msgs.msg import Header

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

MOTION_DT = 0.05  # MotionTarget period, s of sim time (20 Hz, S2b; guide_motion.planner.DT)
# a move has stalled: STALL_S s of sim time without the tool closing STALL_M or STALL_RAD on the goal
STALL_S, STALL_M, STALL_RAD = 0.5, 0.001, np.radians(0.5)


def _oscbf_failures():
    """What controller: oscbf answers with False and a log line, never a raise (spec 8). Imported
    here, not at the top: MoveIt mode never loads guide_motion."""
    from guide_motion.planner import PlanningError
    from pink.exceptions import NoSolutionFound

    return PlanningError, NoSolutionFound, ValueError, TimeoutError


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
        if self.config.controller not in ("moveit", "oscbf"):  # anything else would silently mean MoveIt
            raise ValueError(f"controller must be 'moveit' or 'oscbf', got {self.config.controller!r}")
    
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Robot is already connected.")
    
        # ROS2 initialization
        self._init_ros()
          
        if self.config.controller == "oscbf":
            self.init_oscbf()  # guide_motion plans and streams; nothing of MoveIt is started
        else:
            # Connect to MoveIt2 and Gripper interfaces based on the specified action types
            if self.config.arm_action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY, ActionType.CARTESIAN_POSE):
                self.init_moveit2()
            if self.config.gripper_action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
                self.init_gripper_interface()
            if self.config.arm_action_type == ActionType.CARTESIAN_VELOCITY:
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
                    obs_dict[f'{joint_name}.effort'] = self.joint_state.effort[idx]
                    
        for joint_name in self.config.gripper_joint_names:
            if joint_name in self.joint_state.name:
                idx = self.joint_state.name.index(joint_name)
                if self.config.gripper_observation_type == ObservationType.JOINT_POSITION:
                    obs_dict[f'{joint_name}.pos'] = self.joint_state.position[idx]
                if self.config.gripper_observation_type == ObservationType.JOINT_VELOCITY:
                    obs_dict[f'{joint_name}.vel'] = self.joint_state.velocity[idx]
                if self.config.gripper_observation_type == ObservationType.JOINT_EFFORT:
                    obs_dict[f'{joint_name}.effort'] = self.joint_state.effort[idx]
                    
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
        
            

    def send_action(self, action: Pose | Twist | dict[str, float], cartesian=False, wait_for_execution=False,
                    speed: float | None = None, min_fraction: float = 0.0) -> bool:
        """speed: GUIDE-EX's speed, 0 to 1: MoveIt's velocity scaling (None keeps the current one), or
        controller: oscbf's cruise speed, speed x 0.5 m/s (None = 1.0). min_fraction: MoveIt's; a
        Cartesian path MoveIt can compute for less than this share of the way is refused before the
        arm moves (controller: oscbf plans straight lines or refuses)."""
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is not connected.")
        if self.config.controller == "oscbf":
            try:
                return self._send_oscbf(action, wait_for_execution, speed)
            except _oscbf_failures() as error:
                self.node.get_logger().error(f"guide_motion planner refused the action: {type(error).__name__}: {error}")
                return False
        if speed is not None and self._moveit2 is not None:
            self._moveit2.max_velocity = speed
            
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
                            # See the CARTESIAN_POSE branch: propagate the execution
                            # result so a STATUS_ABORTED trajectory counts as failure.
                            action_executed = bool(self._moveit2.wait_until_executed())
                            if not action_executed:
                                self.node.get_logger().error(
                                    "Joint move aborted; MoveIt execution error_code="
                                    f"{self._moveit2.get_last_execution_error_code()}"
                                )
                    else:
                        self.node.get_logger().warn("Incomplete arm joint configuration provided in action.")
                
                elif self.config.arm_action_type == ActionType.JOINT_TRAJECTORY and arm_goal_pos:
                    self.node.get_logger().warn("JOINT_TRAJECTORY action type is not yet implemented.")
                    
        elif self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            # MoveIt executes whatever part of a Cartesian path it could compute and reports success
            # (pymoveit2's threshold defaults to 0): stopping short of a cube, or at the edge of a
            # self-collision no later plan can start from. Ask for the path first.
            if isinstance(action, Pose) and cartesian and min_fraction > 0:
                future = self._moveit2.plan_async(
                    pose=action, frame_id=self.config.frame_id, cartesian=True, start_joint_state=self.joint_state)
                while future is not None and not future.done():
                    spin_or_wait(self.node)  # MoveIt2.plan's wait; plan() itself keeps the fraction to itself
                if future is None or self._moveit2.get_trajectory(
                        future, cartesian=True, cartesian_fraction_threshold=min_fraction) is None:
                    p, fraction = action.position, future.result().fraction if future is not None else 0.0
                    self.node.get_logger().warning(
                        f"No straight path to x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}: MoveIt computed "
                        f"{fraction:.0%} of it (< {min_fraction:.0%}); not executed.")
                    return False
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
                    # Fold the execution result in: wait_until_executed() returns
                    # False on STATUS_ABORTED. Discarding it made an aborted move
                    # (e.g. the retreat) look successful, so the arm continued into
                    # the bin at grasp height. Now an aborted trajectory is a failure.
                    action_executed = bool(self._moveit2.wait_until_executed())
                    if not action_executed:
                        self.node.get_logger().error(
                            "Cartesian move aborted; MoveIt execution error_code="
                            f"{self._moveit2.get_last_execution_error_code()}"
                        )
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
                        self._gripper_interface.move_to_position(
                            gripper_goal_vals[0], start_joint_state=self.joint_state
                        )
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
        
        if getattr(self, "_stream_thread", None) is not None:
            self._stop_stream.set()
            self._stream_thread.join()
            self._stream_thread = None

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

    def sleep(self, seconds: float, stall_timeout: float = 30.0) -> None:
        """Block for `seconds` of the node's clock: SIM time under use_sim_time.

        GUIDE-EX's waits use it, so a wait lasts the same simulated time at any RTF. Raises
        when the clock stops advancing for `stall_timeout` s of wall time (a dead simulator).
        """
        clock = self.node.get_clock()
        last = clock.now()
        end = last + Duration(seconds=seconds)
        wall = time.perf_counter()
        while (now := clock.now()) < end:
            if now < last:  # the simulator restarted, and its clock with it: keep the time still to wait
                end = now + (end - last)
            if now != last:
                last, wall = now, time.perf_counter()
            elif time.perf_counter() - wall > stall_timeout:
                raise TimeoutError(f"The clock did not advance for {stall_timeout} s of wall time; is the simulation running?")
            time.sleep(0.002)

    def callService(
        self,
        service: Client,
        request,
        message: str | None = None,
        timeout_sec: float = 60.0,
        retries: int = 1,
    ):
        if isinstance(message, str):
            self.node.get_logger().info(message)
        self.node.get_logger().info(f"Calling {service.srv_name} service.")

        # The node is spun continuously by the background MultiThreadedExecutor
        # (_ros_thread) and the clients live in a ReentrantCallbackGroup, so we wait
        # on the async future from this worker thread instead of spinning a second
        # executor. That removes the nested-service deadlock (spinning a node that is
        # already being spun) the old @executor_safe juggling worked around, and the
        # timeout means a single dropped/late reply cannot wedge the whole run.
        last_error: Exception = TimeoutError(f"Service '{service.srv_name}' never responded.")
        for attempt in range(retries + 1):
            future = service.call_async(request)
            done = Event()
            future.add_done_callback(lambda _f: done.set())  # fires immediately if already done

            if done.wait(timeout_sec):
                self.node.get_logger().info(f"Called {service.srv_name} successfully.")
                return future.result()

            last_error = TimeoutError(
                f"Service '{service.srv_name}' timed out after {timeout_sec}s"
            )
            self.node.get_logger().warn(f"{last_error} (attempt {attempt + 1}/{retries + 1}).")
            try:
                service.remove_pending_request(future)  # avoid leaking the pending request
            except Exception:
                pass

        raise last_error

    # ------------------------------------------------------------------
    # controller: oscbf (S2b): guide_motion plans, MotionTargets stream at 20 Hz of sim time
    # to the OSCBF controller that runs in the simulator's (or the robot's) control loop.
    # ------------------------------------------------------------------
    def init_oscbf(self):
        from guide_motion.model import Model
        from guide_msgs.msg import MotionTarget

        self._motion = Model.load(self.config.motion_robot)
        assert self._motion.arm_joints == list(self.config.arm_joint_names), self._motion.arm_joints
        self._motion_msg = MotionTarget
        self._motion_pub = self.node.create_publisher(MotionTarget, "motion_target", 1)
        self._queue, self._last, self._width = deque(), None, float("nan")
        self._queue_lock, self._stop_stream = Lock(), Event()
        self._stream_thread = Thread(target=self._stream, daemon=True)
        self._stream_thread.start()

    def _stream(self):
        """Every 50 ms of sim time: the next planned target, else the last pose/joint target
        again, so the controller never goes stale while it holds a goal."""
        from guide_motion.targets import TWIST

        clock, period = self.node.get_clock(), int(MOTION_DT * 1e9)
        due = clock.now().nanoseconds
        while not self._stop_stream.wait(0.002):  # poll the clock every 2 ms of wall time
            now = clock.now().nanoseconds
            if not due - period <= now < due + period:  # the clock restarted, or ran a period past due: re-anchor
                due = now
            if now < due:
                continue
            due += period  # from the last due, not from now: 20 Hz whatever the frame length divides
            with self._queue_lock:
                if self._queue:
                    self._last = self._queue.popleft()
                target, width = self._last, self._width
                if target is not None and target.mode == TWIST:
                    self._last = None  # one-shot, like a servo command: the host holds the pose after 0.25 s
                if target is not None:  # under the lock: once hold_off() returns, nothing it cleared goes out
                    self._motion_pub.publish(self._motion_target_msg(target, width))

    def _motion_target_msg(self, target, width):
        msg = self._motion_msg()
        msg.stamp = self.node.get_clock().now().to_msg()
        msg.mode = int(target.mode)
        msg.frame_id = self._motion.base_link
        x, y, z, w = Rotation.from_matrix(target.rotation).as_quat()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in target.position)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = float(x), float(y), float(z), float(w)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (float(v) for v in target.twist[:3])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (float(v) for v in target.twist[3:])
        msg.joints = [float(v) for v in target.joints]
        msg.gripper_width = float(width)  # always set: the field's default 0.0 closes the hand; NaN leaves it
        return msg

    def _send_oscbf(self, action, wait_for_execution: bool, speed: float | None) -> bool:
        from guide_motion.planner import plan
        from guide_motion.targets import JOINTS, TWIST, Target

        if self._motion_pub.get_subscription_count() == 0:  # every target would vanish, each try timing out
            self.node.get_logger().error(f"no OSCBF host subscribes to {self._motion_pub.topic_name}: "
                                         "is GUIDE running with controller: oscbf for this scene?")
            return False
        names, fingers, goal = list(self.config.arm_joint_names), [], None
        if isinstance(action, Pose):
            # Ruling 4: poses arrive in the robot's base frame; GUIDE converts, the solver looks nothing up
            if self.config.frame_id != self._motion.base_link:
                self.node.get_logger().error(
                    f"Pose in '{self.config.frame_id}', not the base frame '{self._motion.base_link}': refused.")
                return False
            p, o = action.position, action.orientation
            goal = Target(position=np.array([p.x, p.y, p.z], float), rotation=Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())
        elif isinstance(action, Twist):
            v, w = action.linear, action.angular
            goal = Target(mode=TWIST, twist=np.array([v.x, v.y, v.z, w.x, w.y, w.z], float))
        elif isinstance(action, dict):
            fingers = [v for k, v in action.items() if k.removesuffix(".pos") in self.config.gripper_joint_names]
            arm = {k.removesuffix(".pos"): v for k, v in action.items() if k.removesuffix(".pos") in names}
            if arm and len(arm) != len(names):
                self.node.get_logger().warn("Incomplete arm joint configuration provided in action.")
                return False
            if arm:
                goal = Target(mode=JOINTS, joints=np.array([arm[j] for j in names], float))
            elif not fingers:
                return False
        else:
            return False
        q = self._arm_state()[0]
        # Ruling 3: GUIDE-EX's speed sets the cruise speed, speed x 0.5 m/s; None = 1.0
        targets = [] if goal is None else plan(self._motion, goal, q, speed=1.0 if speed is None else speed)
        with self._queue_lock:  # the arm goal is planned (or there is none): only now the gripper
            if fingers:
                self._width = 2.0 * float(fingers[0])  # one finger's travel -> opening width
                if not targets and self._last is None and not self._queue:  # nothing streams yet: hold here so the width goes out
                    position, rotation = self._motion.ee_pose(q)
                    self._last = Target(position=position, rotation=rotation)
            if targets:
                self._queue = deque(targets)
        if goal is None or not wait_for_execution or goal.mode == TWIST:
            return True
        return self._wait_reached(goal, timeout=2.0 + 2.0 * len(targets) * MOTION_DT, plan_s=len(targets) * MOTION_DT)

    def hold_off(self) -> None:
        """Stop streaming: nothing queued, nothing repeated (Plan B final review I2). GUIDE's Reset
        homes the arm through the host until the next fresh target, and the 20 Hz repeat of the last
        goal would be one within 50 ms. The next send_action resumes the stream."""
        with self._queue_lock:
            self._queue, self._last = deque(), None

    def reset_scene(self, service, request, home) -> bool:
        """GUIDE's Reset under controller: oscbf, the one Reset path: hold off (False, nothing changed,
        if the planner refuses `home`), call `service` (GUIDE's Reset), then wait on sim time until the tool is within the reach tolerance of
        `home` (the scene's default_joint_states[:7], where the host homes the arm), for at most
        2 s + 2 x the homing plan's duration (the host plans it at speed 1.0)."""
        from guide_motion.planner import plan
        from guide_motion.targets import JOINTS, Target

        goal = Target(mode=JOINTS, joints=np.asarray(home, float))
        try:
            homing = plan(self._motion, goal, self._arm_state()[0], speed=1.0)  # before hold_off: a refused home changes nothing
            self.hold_off()
            response = self.callService(service, request, "Resetting the scene")
            if not response.success:
                self.node.get_logger().error(f"Reset failed: {response.message}")
                return False
            return self._wait_reached(goal, timeout=2.0 + 2.0 * len(homing) * MOTION_DT, plan_s=len(homing) * MOTION_DT)
        except _oscbf_failures() as error:
            self.node.get_logger().error(f"reset_scene: {type(error).__name__}: {error}")
            return False

    def _arm_state(self):
        js = self.joint_state
        index = [js.name.index(j) for j in self.config.arm_joint_names]
        q = np.array([js.position[i] for i in index], float)
        qd = np.array([js.velocity[i] for i in index], float) if len(js.velocity) == len(js.name) else np.zeros(len(index))
        return q, qd

    def _wait_reached(self, goal, timeout: float, stall_timeout: float = 30.0, plan_s: float | None = None) -> bool:
        """Tool within 5 mm and 2 deg of the goal, slower than 1 cm/s (a JOINTS goal: every joint within
        0.01 rad as well), held 0.1 s of sim time (S2b 4.5). Given the plan's duration `plan_s`, an arm
        that has not arrived and makes no progress (STALL_S without closing STALL_M or STALL_RAD) once
        the plan has run out has stalled -- on a contact, a barrier; it may still jitter -- and fails
        then instead of at `timeout`."""
        from guide_motion.targets import JOINTS

        position, rotation = self._motion.ee_pose(goal.joints) if goal.mode == JOINTS else (goal.position, goal.rotation)
        clock = self.node.get_clock()
        start = last = clock.now()
        wall, held, mark = time.perf_counter(), None, None
        while (now := clock.now()) - start < Duration(seconds=timeout):
            if now < start:  # the simulator restarted, and its clock with it
                start, held, mark = now, None, None
            if now != last:
                last, wall = now, time.perf_counter()
            elif time.perf_counter() - wall > stall_timeout:
                message = f"The clock did not advance for {stall_timeout} s of wall time; is the simulation running?"
                break
            q, qd = self._arm_state()
            p, R = self._motion.ee_pose(q)
            dist, angle = np.linalg.norm(p - position), Rotation.from_matrix(rotation.T @ R).magnitude()
            reached = (dist < 0.005
                       and angle < np.radians(2.0)
                       and self._motion.ee_speed(q, qd) < 0.01
                       and (goal.mode != JOINTS or np.max(np.abs(q - goal.joints)) < 0.01))  # a joint goal: the joints too
            held = (held or now) if reached else None
            if held is not None and now - held >= Duration(seconds=0.1):
                return True
            if plan_s is None or reached or now - start < Duration(seconds=plan_s):
                mark = None
            elif mark is None or dist < mark[1] - STALL_M or angle < mark[2] - STALL_RAD:
                mark = (now, dist, angle)  # progress: the window starts again
            elif now - mark[0] >= Duration(seconds=STALL_S):
                message = self._missed("stalled", (now - start).nanoseconds * 1e-9, position, rotation)
                break
            time.sleep(0.002)
        else:
            message = self._missed("timed out", timeout, position, rotation)
        self.node.get_logger().error(message)
        self.hold_off()  # False means stopped: the host holds the latched pose, not the goal the caller was told failed
        return False

    def _missed(self, verb: str, seconds: float, position, rotation) -> str:
        """How far the tool is from the goal, and the joints, for a move that failed."""
        q, _ = self._arm_state()
        p, R = self._motion.ee_pose(q)
        return (f"OSCBF move {verb} after {seconds:.1f} s of sim time, {1000 * np.linalg.norm(p - position):.1f} mm and "
                f"{np.degrees(Rotation.from_matrix(rotation.T @ R).magnitude()):.1f} deg from the goal "
                f"at q={np.round(q, 4).tolist()}")
