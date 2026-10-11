import cv2
import numpy as np

from threading import Thread
import time

from lerobot.cameras import Camera

import rclpy

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from .config import ROS2CameraConfig


def _imgmsg_to_bgr(msg) -> np.ndarray:
    """cv_bridge-free ROS sensor_msgs/Image -> HxWx3 uint8 BGR.

    cv_bridge is built against numpy<2 and cannot import in the numpy-2 Isaac
    venv, so decode 8-bit colour images directly. Output is BGR to preserve the
    original ``desired_encoding='bgr8'`` behaviour.
    """
    channels = max(1, msg.step // max(1, msg.width))
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    arr = arr[:, :, :3]
    if msg.encoding.startswith("rgb"):
        arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(arr)


#: ROS depth encodings this class understands, and the dtype each one packs.
_DEPTH_DTYPES = {"32FC1": np.float32, "16UC1": np.uint16}


def _depthmsg_to_uint16_mm(msg) -> np.ndarray:
    """ROS depth image -> (H, W, 1) uint16 millimetres.

    Isaac's ROS2CameraHelper publishes 32FC1 -- float32 metres, +inf where the ray hit
    nothing. A RealSense publishes 16UC1, already millimetres. Both land in the same
    array, so a policy cannot tell sim from hardware by dtype, and it is the same shape
    the recorder writes into the dataset (guide_core's ``depth_to_uint16_mm``): lerobot
    flags a 1-channel camera as ``is_depth_map`` and reads a non-floating dtype as
    millimetres.

    Misses become 0 rather than saturating to 65.5 m, which would otherwise put a wall
    of maximum-range readings wherever the sky is.
    """
    dtype = _DEPTH_DTYPES.get(msg.encoding)
    if dtype is None:
        raise ValueError(
            f"Unsupported depth encoding '{msg.encoding}'; expected one of "
            f"{sorted(_DEPTH_DTYPES)}."
        )
    raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    if dtype is np.float32:
        raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0.0, 65535.0)
    return raw.astype(np.uint16).reshape(msg.height, msg.width, 1)
    
    
class ROS2Camera(Camera):
    
    def __init__(self, config: ROS2CameraConfig):
        super().__init__(config)
        
        self.namespace: str = config.namespace
        self.topic: str = config.topic
        self.frame_id: str = config.frame_id
        # One camera object per stream: an rgb camera and a depth camera on the same
        # physical camera are two ROS2Camera instances on two topics, because lerobot
        # keys one dataset feature per Camera.
        self.depth: bool = config.depth
        
        self.node: Node = None
        self.image: np.ndarray = None
        self._decoder = None
        
        self.original_width = None
        self.original_height = None
        
        self.start = None
        

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
            # The image stream gets its own mutually exclusive group. A ROS 2 callback
            # in a reentrant group runs concurrently with ITSELF on a
            # MultiThreadedExecutor, and the H.264 decoder is one FFmpeg context per
            # camera -- four threads went into _cameraCompressedCallback at once and
            # corrupted the heap ("malloc(): unsorted double linked list corrupted",
            # SIGABRT, before the first episode). Decoding is stateful anyway: H.264 is
            # inter-frame, so packets have to be fed in order, which concurrent
            # callbacks cannot promise even without the crash.
            self._image_callback_group = MutuallyExclusiveCallbackGroup()
            
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
        
        if self.depth:
            message_type, topic, callback = Image, f"{self.topic}/depth", self._cameraDepthCallback
        else:
            message_type, topic = self._discover_rgb_stream()
            callback = (
                self._cameraImageCallback
                if message_type is Image
                else self._cameraCompressedCallback
            )
            if message_type is CompressedImage:
                import av  # only needed for an H.264 stream; a raw deployment has no PyAV

                self._decoder = av.CodecContext.create("h264", "r")

        self.node.get_logger().info(
            f"Subscribing to {self.node.resolve_topic_name(topic)} "
            f"({message_type.__name__})."
        )
        self.camera_image_sub = self.node.create_subscription(
            message_type,
            topic,
            callback,
            1,
            callback_group=self._image_callback_group
        )
        
        if warmup:
            self._wait_for_image(timeout=5000)

    def _discover_rgb_stream(self, timeout: float = 5.0):
        """Ask the ROS graph which rgb stream the camera helper actually created.

        Isaac's ROS2CameraHelper publishes either ``sensor_msgs/Image`` on ``<topic>``
        or ``sensor_msgs/CompressedImage`` on ``<topic>/compressed``, decided by
        ``camera_encoding`` in the task's ``config/init.yaml``. Reading it off the graph
        rather than off a second copy of that setting means the two can never disagree
        -- a mismatch would not raise, it would subscribe to a name nothing publishes
        and time out in warmup -- and a real camera, which publishes raw, needs no
        configuration at all.

        Falls back to raw after ``timeout``: no publisher yet is the same situation a
        subscriber is in before the simulator has built its graphs, and
        ``_wait_for_image`` already reports that case properly.
        """
        candidates = ((CompressedImage, f"{self.topic}/compressed"), (Image, self.topic))
        deadline = time.time() + timeout
        while rclpy.ok():
            for message_type, topic in candidates:
                if self.node.get_publishers_info_by_topic(self.node.resolve_topic_name(topic)):
                    return message_type, topic
            if time.time() >= deadline:
                break
            time.sleep(0.05)

        self.node.get_logger().warn(
            f"No publisher on {self.node.resolve_topic_name(self.topic)} or its "
            f"/compressed variant after {timeout:.0f}s; assuming raw sensor_msgs/Image."
        )
        return Image, self.topic
            
    def _wait_for_image(self, timeout: float = None) -> bool:
        # `timeout` is in milliseconds; None waits indefinitely. Bounded so warmup /
        # read never block forever when the camera publisher is disabled (scene config
        # flag publish_camera_topics) and no image ever arrives.
        start = time.time()
        warned = False
        while self.image is None and rclpy.ok() and self.node is not None:
            elapsed_ms = (time.time() - start) * 1000.0
            if timeout is not None and elapsed_ms >= timeout:
                self.node.get_logger().warn(
                    f"No image on {self.namespace}/{self.topic} after {timeout:.0f}ms; "
                    f"continuing without it (publisher may be disabled)."
                )
                return False
            if not warned and elapsed_ms > 1000.0:
                self.node.get_logger().warn(
                    f"Waiting for {self.namespace}/{self.topic} first message..."
                )
                warned = True
            time.sleep(0.01)

        return self.image is not None
        
    def disconnect(self):
        self.start.set_result(True)
        
        self.ros_thread.join()
        
        self.node = None
        self.start = None
        
        self.image = None
        self._decoder = None
        
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
        self._set_image(cv2.cvtColor(_imgmsg_to_bgr(msg), cv2.COLOR_BGR2RGB))

    def _cameraCompressedCallback(self, msg: CompressedImage):
        # H.264 is inter-frame: packets must be fed in order and decoding only starts at
        # the first IDR, so a subscriber that joins mid-episode stays blank until the
        # encoder's next keyframe rather than showing a torn frame. That wait is what
        # _wait_for_image absorbs. parse() may also yield nothing for a packet that is
        # only part of a frame, which is why nothing here assumes one message in, one
        # image out.
        for packet in self._decoder.parse(bytes(msg.data)):
            for frame in self._decoder.decode(packet):
                self._set_image(frame.to_ndarray(format="rgb24"))

    def _cameraDepthCallback(self, msg: Image):
        self._set_image(_depthmsg_to_uint16_mm(msg))

    def _set_image(self, image: np.ndarray) -> None:
        # Against the array, not against camera_info: `original_*` are ints read off a
        # message, and the old `original_height is not self.height` compared them by
        # identity, which is False for any int above CPython's small-int cache. Every
        # frame took the resize path, and a camera whose camera_info never arrived
        # compared against None. The frame itself always knows its own size.
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = self._resize_keep_max_area(image, self.width, self.height)

        self.image = image
        
    def _resize_keep_max_area(self, img, target_width=640, target_height=480):
        h, w = img.shape[:2]

        # Scaling factors
        scale = max(target_width / w, target_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resizing. Depth is nearest-neighbour: averaging across a depth discontinuity
        # invents a surface halfway between the foreground and the background.
        interpolation = cv2.INTER_NEAREST if self.depth else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        if resized.ndim == 2 and img.ndim == 3:
            resized = resized[:, :, np.newaxis]  # cv2 drops a trailing 1-channel axis

        x_start = (new_w - target_width) // 2
        y_start = (new_h - target_height) // 2

        cropped = resized[y_start : y_start + target_height, x_start : x_start + target_width]

        return cropped
        
    @staticmethod
    def find_cameras():
        return []
    
    def read(self, color_mode = None):
        if self.image is None:
            self._wait_for_image(timeout=5000)
            
        if color_mode is not None and not self.depth:
            return cv2.cvtColor(self.image, color_mode)
        
        return self.image
    
    def async_read(self, timeout_ms = ...):
        if self.image is None:
            self._wait_for_image(timeout=timeout_ms)
            
        return self.image