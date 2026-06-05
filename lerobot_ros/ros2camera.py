import cv2
import numpy as np

from cv_bridge import CvBridge
from threading import Thread
import time

from lerobot.cameras import Camera

import rclpy

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from .config import ROS2CameraConfig
    
    
class ROS2Camera(Camera):
    
    def __init__(self, config: ROS2CameraConfig):
        super().__init__(config)
        
        self.namespace: str = config.namespace
        self.topic: str = config.topic
        self.frame_id: str = config.frame_id
        
        self.node: Node = None
        self.image: np.ndarray = None
        
        self.original_width = None
        self.original_height = None
        
        self.start = None
        self.bridge = CvBridge()
        

    def _init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        if self.node is None:
            self.node = Node(
                node_name=self.frame_id, 
                namespace=self.namespace,
                parameter_overrides=[
                    rclpy.Parameter(name='use_sim_time', value=True)
                ]
            )
            self._reentrant_callback_group = ReentrantCallbackGroup()
            
            self.start = Future()
            self.ros_thread = Thread(target=self._ros_thread)
            self.ros_thread.start()
            
    def _ros_thread(self):
        if self.node is not None:
            self.executor: MultiThreadedExecutor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.executor.spin_until_future_complete(self.start)
            self.executor.remove_node(self.node)
            self.node.destroy_node()
        

    def connect(self, warmup: bool = True):
        self._init_ros()
        
        self.camera_info_sub = self.node.create_subscription(
            CameraInfo,
            'camera_info',
            self._cameraInfoCallback,
            1,
            callback_group=self._reentrant_callback_group
        )
        
        self.camera_image_sub = self.node.create_subscription(
            Image,
            self.topic,
            self._cameraImageCallback,
            1,
            callback_group=self._reentrant_callback_group
        )
        
        if warmup:
            self._wait_for_image()
            
    def _wait_for_image(self, timeout: float = None) -> bool:
        rate = self.node.create_rate(0.01, self.node.get_clock())
        count = 0
        while self.image is None and rclpy.ok() and self.node is not None:
            if count%100 == 1:
                self.node.get_logger().warn(f'Waiting for {self.namespace}/{self.topic} for first message.') 
            time.sleep(0.01)
            count += 1
            if timeout is not None and timeout >= count*10:
                break
        # Cleanup
        rate.destroy()
        
        return self.image is not None
        
    def disconnect(self):
        self.start.set_result(True)
        
        self.ros_thread.join()
        
        self.node = None
        self.start = None
        
        self.image = None
        
    @property
    def is_connected(self):
        return self.node is not None
    
    # TODO: Implement waiting for camera info and distorsion parameters.
    def calibrate(self):
        pass
    
    # TODO: Implement calibrated logic.
    def is_calibrated(self):
        return True
    
    def _cameraInfoCallback(self, msg: CameraInfo):
        if msg.header.frame_id is not self.frame_id:
            return
        
        self.original_width = msg.width
        self.original_height = msg.height
     
    # TODO: Implement camera correction for distorsion.
    def _cameraImageCallback(self, msg: Image):
        image = cv2.cvtColor(
                    self.bridge.imgmsg_to_cv2(
                        msg, desired_encoding="bgr8"
                    ),
                    cv2.COLOR_BGR2RGB,
                )
        
        if self.original_height is not self.height or self.original_width is not self.width:
            image = self._resize_keep_max_area(image, self.width, self.height)
        
        self.image = image
        
    def _resize_keep_max_area(self, img, target_width=640, target_height=480):
        h, w = img.shape[:2]

        # Scaling factors
        scale = max(target_width / w, target_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resizing
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_start = (new_w - target_width) // 2
        y_start = (new_h - target_height) // 2

        cropped = resized[y_start : y_start + target_height, x_start : x_start + target_width]

        return cropped
        
    @staticmethod
    def find_cameras():
        return []
    
    def read(self, color_mode = None):
        if self.image is None:
            self._wait_for_image()
            
        if color_mode is not None:
            return cv2.cvtColor(self.image, color_mode)
        
        return self.image
    
    def async_read(self, timeout_ms = ...):
        if self.image is None:
            self._wait_for_image(timeout=timeout_ms)
            
        return self.image