"""ROS2Robot's controller: oscbf mode and the sim-time sleep, against a fake node and clock."""

import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Pose, Twist
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irob_lerobot_ros.config import ActionType, ObservationType  # noqa: E402
from irob_lerobot_ros.ros2robot import ROS2Robot  # noqa: E402

ARM = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]


class FakeClock:
    """Sim time that advances `step` s on every read, so the polling loops make progress."""

    def __init__(self, step=0.001):
        self.t, self.step = 0, int(step * 1e9)

    def now(self):
        self.t += self.step
        return Time(nanoseconds=self.t)


class RestartingClock(FakeClock):
    """Restarts at 0 once, `after` s past `start`, as Isaac's clock does on stop/play."""

    def __init__(self, step=0.001, start=0.0, after=0.05):
        super().__init__(step)
        self.t, self.restart_at, self.restarted = int(start * 1e9), int((start + after) * 1e9), False

    def now(self):
        if self.t > self.restart_at and not self.restarted:
            self.t, self.restarted = 0, True
        return super().now()


def robot(q=HOME, clock=None, controller="oscbf"):
    from guide_motion.model import Model

    r = ROS2Robot.__new__(ROS2Robot)
    r.logs = []
    log = r.logs.append
    clock = clock or FakeClock()
    r.node = SimpleNamespace(get_clock=lambda: clock, get_logger=lambda: SimpleNamespace(error=log, warn=log, warning=log, info=log))
    r.cameras = {}
    r.config = SimpleNamespace(controller=controller, arm_joint_names=ARM, gripper_joint_names=["fr3_finger_joint1"],
                               frame_id="fr3_link0", arm_action_type=ActionType.CARTESIAN_POSE)
    r.joint_state = JointState(name=ARM, position=list(q), velocity=[0.0] * 7)
    r._motion = Model.load("fr3")
    r._queue, r._last, r._width, r._queue_lock = deque(), None, float("nan"), threading.Lock()
    return r


@pytest.fixture
def make():
    """robot(...) factory; drops the fake node afterwards, or __del__ would 'disconnect' it."""
    made = []
    yield lambda **kwargs: made.append(robot(**kwargs)) or made[-1]
    for r in made:
        r.node = None


def pose(position, rotation):
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = float(x), float(y), float(z), float(w)
    return p


def test_sleep_counts_sim_time_not_wall_time(make):
    r = make(clock=FakeClock(step=0.01))
    start = time.perf_counter()
    r.sleep(1.0)
    assert r.node.get_clock().t >= 1e9 and time.perf_counter() - start < 1.0


def test_sleep_raises_when_the_clock_stops(make):
    r = make(clock=FakeClock(step=0.0))
    with pytest.raises(TimeoutError):
        r.sleep(1.0, stall_timeout=0.05)


def test_sleep_keeps_the_remaining_time_when_the_clock_restarts(make):
    clock = RestartingClock(step=0.01, start=20.0, after=0.3)  # a RegisterScene's stop/play 0.3 s into the wait
    r = make(clock=clock)
    r.sleep(1.0)
    assert clock.restarted and 0.70e9 <= clock.t <= 0.75e9  # 0.3 s before the restart + 0.7 s after: 1 s of sim time


def test_moveit_refuses_a_cartesian_path_it_cannot_finish(make):
    r = make(controller="moveit")
    moved = []
    r._moveit2 = SimpleNamespace(max_velocity=1.0, plan=lambda **kwargs: None, move_to_pose=lambda **kwargs: moved.append(kwargs) or True)
    assert not r.send_action(Pose(), cartesian=True, wait_for_execution=True, speed=0.2, min_fraction=0.95)
    assert moved == [] and r._moveit2.max_velocity == 0.2


def test_observation_features_include_the_gripper(make):
    r = make()
    r.config.arm_observation_type = ObservationType.JOINT_POSITION
    r.config.gripper_observation_type = ObservationType.JOINT_POSITION
    assert list(r._motors_obs_ft) == [f"{j}.pos" for j in ARM] + ["fr3_finger_joint1.pos"]


@pytest.mark.parametrize("kind", list(ObservationType))
def test_observations_use_the_declared_feature_keys(make, kind):
    r = make()
    r.config.arm_observation_type = r.config.gripper_observation_type = kind
    r.joint_state = JointState(name=ARM + ["fr3_finger_joint1"], position=[0.0] * 8, velocity=[0.0] * 8, effort=[0.0] * 8)
    assert list(r.get_observation()) == list(r._motors_obs_ft)
