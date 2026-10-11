# flake8: noqa: E402
"""A depth message has to arrive as the array the dataset already stores.

The recorder writes (H, W, 1) uint16 millimetres into the LeRobot dataset, because that
is the shape ``hw_to_dataset_features`` flags as ``is_depth_map`` and the unit
``quantize_depth`` reads off a non-floating dtype. A policy reading the live topic has
to be handed the same thing, or it sees metres at training time and millimetres at
inference -- a 1000x scale error that no shape check would catch.

Isaac publishes 32FC1 (float32 metres, +inf for a miss); a RealSense publishes 16UC1
(already millimetres). Both land in the same array.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def ros2camera():
    """``irob_lerobot_ros.ros2camera`` with the ROS runtime stubbed out."""
    for name in (
        "rclpy",
        "rclpy.callback_groups",
        "rclpy.executors",
        "rclpy.node",
        "rclpy.task",
        "sensor_msgs",
        "sensor_msgs.msg",
    ):
        sys.modules.setdefault(name, MagicMock())

    import irob_lerobot_ros.ros2camera as module

    return module


def depth_message(array, encoding):
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        data=array.tobytes(),
    )


def test_isaac_metres_become_millimetres(ros2camera):
    metres = np.array([[0.0, 1.5], [0.001, 12.0]], dtype=np.float32)

    depth = ros2camera._depthmsg_to_uint16_mm(depth_message(metres, "32FC1"))

    assert depth.dtype == np.uint16
    assert depth.shape == (2, 2, 1)
    assert depth[..., 0].tolist() == [[0, 1500], [1, 12000]]


def test_a_realsense_is_already_in_millimetres(ros2camera):
    # 16UC1 must pass through untouched. Multiplying it by 1000 as well would be the
    # same 1000x error in the other direction, and both dtypes are legal depth images.
    millimetres = np.array([[0, 1500], [1, 12000]], dtype=np.uint16)

    depth = ros2camera._depthmsg_to_uint16_mm(depth_message(millimetres, "16UC1"))

    assert depth[..., 0].tolist() == [[0, 1500], [1, 12000]]


def test_misses_read_as_zero_not_as_maximum_range(ros2camera):
    metres = np.array([[np.inf, np.nan, 100.0]], dtype=np.float32)

    depth = ros2camera._depthmsg_to_uint16_mm(depth_message(metres, "32FC1"))

    assert depth[..., 0].tolist() == [[0, 0, 65535]]


def test_an_unknown_encoding_is_an_error_not_a_reinterpret_cast(ros2camera):
    # Guessing here would reshape whatever bytes arrived into a plausible-looking depth
    # map: silently wrong data is worse than a stopped run.
    message = depth_message(np.zeros((2, 2), dtype=np.float32), "mono8")

    with pytest.raises(ValueError, match="mono8"):
        ros2camera._depthmsg_to_uint16_mm(message)
