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
    r._motion_pub = SimpleNamespace(publish=lambda msg: None, get_subscription_count=lambda: 1,
                                    topic_name="/Sim_0/Scene_0/franka/motion_target")  # GUIDE's OSCBF host listens
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


def moveit_computing(fraction, moved):
    """A fake MoveIt whose straight-line plan covers `fraction` of the way; records executed moves."""
    planned = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(fraction=fraction))
    return SimpleNamespace(
        max_velocity=1.0,
        plan_async=lambda **kwargs: planned,
        get_trajectory=lambda future, cartesian, cartesian_fraction_threshold: object() if fraction >= cartesian_fraction_threshold else None,
        move_to_pose=lambda **kwargs: moved.append(kwargs) or True,
    )


def test_moveit_refuses_a_cartesian_path_it_cannot_finish(make):
    r = make(controller="moveit")
    moved = []
    r._moveit2 = moveit_computing(0.6, moved)
    assert not r.send_action(Pose(), cartesian=True, wait_for_execution=True, speed=0.2, min_fraction=0.95)
    assert moved == [] and r._moveit2.max_velocity == 0.2


def test_a_refused_straight_line_names_the_pose_and_the_fraction(make):
    r = make(controller="moveit")
    r._moveit2 = moveit_computing(0.6, [])
    target = Pose()
    target.position.x, target.position.y, target.position.z = 0.5, 0.1, 0.112
    assert not r.send_action(target, cartesian=True, min_fraction=0.95)
    assert r.logs[-1] == "No straight path to x=0.500, y=0.100, z=0.112: MoveIt computed 60% of it (< 95%); not executed."


def test_moveit_executes_a_straight_line_it_can_finish(make):
    r = make(controller="moveit")
    r.config.gripper_action_type = ActionType.JOINT_POSITION  # a Pose passes the gripper branch by
    moved = []
    r._moveit2 = moveit_computing(1.0, moved)
    assert r.send_action(Pose(), cartesian=True, min_fraction=0.95)
    assert len(moved) == 1


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


def test_a_reached_goal_must_be_held_for_a_tenth_of_a_second(make):
    from guide_motion.targets import Target

    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert r._wait_reached(Target(position=p, rotation=R), timeout=1.0)
    assert r.node.get_clock().t >= 0.1e9


def test_an_arm_elsewhere_times_out_on_sim_time(make):
    from guide_motion.targets import Target

    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p + [0.1, 0, 0], rotation=R), timeout=0.3)
    assert any("timed out" in str(line) and "mm and" in str(line) and "q=[" in str(line) for line in r.logs)  # names what is left


def test_a_still_arm_short_of_the_goal_fails_once_the_plan_has_run_out(make):
    from guide_motion.targets import Target

    from irob_lerobot_ros.ros2robot import STALL_S

    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p + [0.02, 0, 0], rotation=R), timeout=10.0, plan_s=0.4)
    assert 0.4 + STALL_S <= r.node.get_clock().t * 1e-9 < 0.4 + STALL_S + 0.1  # not at the 10 s timeout
    assert any("stalled" in str(line) and "20.0 mm and" in str(line) for line in r.logs)


def test_an_arm_closing_in_slowly_waits_for_the_timeout(make):
    from guide_motion.targets import Target

    r = make()
    clock, q0 = r.node.get_clock(), np.array(HOME)
    state = r._arm_state
    # the wrist turns 4 deg/s towards a goal 20 deg away: slow, but progress (> 0.5 deg per 0.5 s)
    r._arm_state = lambda: (q0 + [0, 0, 0, 0, 0, 0, np.radians(4.0) * clock.t * 1e-9], state()[1])
    p, R = r._motion.ee_pose(q0 + [0, 0, 0, 0, 0, 0, np.radians(20.0)])
    assert not r._wait_reached(Target(position=p, rotation=R), timeout=1.5, plan_s=0.0)
    assert r.node.get_clock().t >= 1.5e9 and any("timed out" in str(line) for line in r.logs)


def test_a_jittering_arm_that_makes_no_progress_has_stalled(make):
    from guide_motion.targets import Target

    from irob_lerobot_ros.ros2robot import STALL_S

    r = make()
    clock, q0 = r.node.get_clock(), np.array(HOME)
    state = r._arm_state
    # pushing on a contact: the joints shake (0.3 rad/s) but the tool gets no closer
    r._arm_state = lambda: (q0 + [0, 0.002 * np.sin(clock.t * 1e-9 * 150), 0, 0, 0, 0, 0], state()[1] + 0.3)
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p + [0.025, 0, 0], rotation=R), timeout=10.0, plan_s=0.2)
    assert r.node.get_clock().t * 1e-9 < 0.2 + STALL_S + 0.1 and any("stalled" in str(line) for line in r.logs)


def test_an_arm_at_the_goal_still_moving_has_not_arrived(make):
    from guide_motion.targets import Target

    r = make()
    clock, q0 = r.node.get_clock(), np.array(HOME)
    state = r._arm_state
    # joint 2 at 0.05 rad/s through the goal: the tool moves ~2.5 cm/s, inside 5 mm for the whole wait
    r._arm_state = lambda: (q0 + [0, 0.05 * (clock.t * 1e-9 - 0.15), 0, 0, 0, 0, 0], state()[1])
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p, rotation=R), timeout=0.3)


def test_an_arm_chattering_on_a_held_handle_has_arrived(make):
    from guide_motion.targets import Target

    r = make()
    # fingers squeezing a fixed bar: the joint velocities read 5 cm/s at the tool, the tool does not move
    r.joint_state.velocity = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    p, R = r._motion.ee_pose(HOME)
    assert r._wait_reached(Target(position=p, rotation=R), timeout=0.3)


def test_speed_sets_the_cruise_speed_and_none_means_full_speed(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert r.send_action(pose(p + [0.0, 0.2, 0.0], R), wait_for_execution=False)
    assert len(r._queue) == 19  # speed None = 1.0: cruise 0.5 m/s
    np.testing.assert_allclose(r._queue[-1].position, p + [0.0, 0.2, 0.0], atol=1e-9)
    assert r.send_action(pose(p + [0.0, 0.2, 0.0], R), speed=0.2)
    assert len(r._queue) == 48  # cruise 0.1 m/s


def test_an_unreachable_pose_returns_false(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p + [1.5, 0.0, 0.0], R))
    assert any("planner" in str(line) for line in r.logs)


def test_a_finger_position_becomes_the_width_and_starts_a_hold(make):
    r = make()
    assert r.send_action({"fr3_finger_joint1.pos": 0.04})
    assert r._width == pytest.approx(0.08) and r._last is not None
    np.testing.assert_allclose(r._last.position, r._motion.ee_pose(HOME)[0])


def test_the_stream_sends_the_plan_then_repeats_the_last_target(make):
    from guide_motion.targets import TWIST, Target

    r = make()
    sent = []
    r._motion_target_msg = lambda target, width: (target, width)
    r._motion_pub = SimpleNamespace(publish=sent.append)
    r._stop_stream = threading.Event()
    targets = [Target(position=np.full(3, i)) for i in range(3)]
    r._queue = deque(targets)
    thread = threading.Thread(target=r._stream, daemon=True)
    thread.start()
    time.sleep(0.8)
    assert [t for t, _ in sent[:3]] == targets and all(t is targets[-1] for t, _ in sent[3:]) and len(sent) > 4
    assert all(np.isnan(width) for _, width in sent)  # no gripper command yet: every target leaves the hand
    with r._queue_lock:
        r._queue = deque([Target(mode=TWIST, twist=np.ones(6))])
    time.sleep(0.5)
    r._stop_stream.set()
    thread.join()
    twists = [t for t, _ in sent if t.mode == TWIST]
    assert len(twists) == 1 and sent[-1][0].mode == TWIST  # one-shot, nothing after it


def test_twist_is_queued_without_waiting(make):
    r = make()
    twist = Twist()
    twist.linear.x = 0.05
    assert r.send_action(twist, wait_for_execution=True)
    assert r._queue[0].twist[0] == pytest.approx(0.05)


def test_a_clock_restart_during_a_wait_still_times_out(make):
    from guide_motion.targets import Target

    r = make(clock=RestartingClock(start=20.0))  # 20 s in, back to 0: without the resync the wait lasts ~20 s
    p, R = r._motion.ee_pose(HOME)
    started = time.perf_counter()
    assert not r._wait_reached(Target(position=p + [0.1, 0, 0], rotation=R), timeout=0.2)
    assert time.perf_counter() - started < 5.0


def test_a_new_goal_replaces_the_queue_and_starts_from_the_measured_state(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    r.send_action(pose(p + [0.0, 0.2, 0.0], R))
    r.send_action(pose(p + [0.0, 0.0, -0.1], R))
    assert np.allclose(r._queue[-1].position, p + [0.0, 0.0, -0.1])
    assert np.linalg.norm(r._queue[0].position - p) < 0.01  # 4.6 mm from where the arm is; the old plan ended 0.2 m away


def test_a_pose_outside_the_base_frame_is_refused(make):
    r = make()
    r.config.frame_id = "Scene_0"
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p, R))
    assert not r._queue and any("refused" in str(line) for line in r.logs)


def test_a_partial_joint_goal_is_refused(make):
    r = make()
    assert not r.send_action({"fr3_joint1.pos": 0.0})
    assert not r._queue


def test_a_target_becomes_a_motion_target_message(make):
    from guide_msgs.msg import MotionTarget

    from guide_motion.targets import JOINTS, Target

    r = make()
    r._motion_msg = MotionTarget
    msg = r._motion_target_msg(Target(mode=JOINTS, joints=np.array(HOME)), 0.08)
    assert msg.mode == MotionTarget.MODE_JOINTS and msg.frame_id == "fr3_link0"
    assert list(msg.joints) == pytest.approx(HOME) and msg.gripper_width == 0.08
    assert msg.pose.orientation.w == pytest.approx(1.0)  # identity rotation for a joint target
    assert np.isnan(r._motion_target_msg(Target(), float("nan")).gripper_width)  # NaN goes out as NaN, not 0.0


def test_reset_holds_off_the_stream_before_it_calls_the_service(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    r.send_action(pose(p + [0.0, 0.2, 0.0], R))
    r._last = r._queue.popleft()  # mid-plan: one target sent, the rest queued
    seen = []
    r.callService = lambda service, request, message=None: seen.append((len(r._queue), r._last)) or SimpleNamespace(success=True)
    assert r.reset_scene("/Sim_0/Reset", "request", HOME)
    assert seen == [(0, None)]  # when the host starts homing, nothing is queued or repeated
    assert not r._queue and r._last is None  # and nothing streams until the next send_action


def test_reset_waits_on_sim_time_for_the_home_configuration(make):
    away = list(HOME)
    away[0] = 0.5
    r = make(q=away, clock=FakeClock(step=0.01))
    r.callService = lambda service, request, message=None: SimpleNamespace(success=True)
    assert not r.reset_scene("/Sim_0/Reset", "request", HOME)  # the arm never arrives
    assert 1.2e9 <= r.node.get_clock().t <= 1.25e9  # stalled: 14 homing targets x 50 ms, then 0.5 s still, sim time
    r.joint_state.position = list(HOME)  # the host has homed it
    assert r.reset_scene("/Sim_0/Reset", "request", HOME)
    r.joint_state.position = list(away)  # a scene whose default_joint_states is `away`, not the XRDF default
    assert r.reset_scene("/Sim_0/Reset", "request", away)


def test_oscbf_never_reaches_into_moveit(make):
    r = make()
    r._moveit2 = None  # controller: oscbf starts no MoveIt
    p, R = r._motion.ee_pose(HOME)
    assert r.send_action(pose(p + [0.0, 0.0, -0.1], R), cartesian=True, speed=0.5, min_fraction=0.95)
    assert len(r._queue) > 1


def test_hold_off_waits_for_a_target_in_flight(make):
    """Plan B I2: once hold_off() returns, nothing it cleared reaches the host, not even a target
    the stream had already popped and was publishing (tester B, C2)."""
    from guide_motion.targets import Target

    r = make()
    publishing, release, sent = threading.Event(), threading.Event(), []
    r._motion_target_msg = lambda target, width: target
    r._motion_pub = SimpleNamespace(publish=lambda msg: publishing.set() or release.wait(1.0) and sent.append(msg))
    r._stop_stream = threading.Event()
    r._queue = deque([Target()])
    thread = threading.Thread(target=r._stream, daemon=True)
    thread.start()
    assert publishing.wait(1.0)  # popped, mid-publish
    threading.Timer(0.1, release.set).start()
    r.hold_off()
    before = len(sent)
    time.sleep(0.2)
    r._stop_stream.set()
    thread.join()
    assert len(sent) == before


def test_reset_to_a_home_outside_the_limits_returns_false_and_keeps_the_stream(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    r.send_action(pose(p + [0.0, 0.2, 0.0], R))
    queued = len(r._queue)
    r.callService = lambda service, request, message=None: pytest.fail("no Reset without a homing plan")
    home = list(HOME)
    home[0] = 10.0  # rad: beyond fr3_joint1's limit
    assert not r.reset_scene("/Sim_0/Reset", "request", home)
    assert any("PlanningError" in str(line) for line in r.logs) and len(r._queue) == queued


def test_a_pose_move_waits_on_sim_time_and_holds_off_when_it_times_out(make):
    """Review C2 I2b + M3: wait_for_execution=True waits for the plan (15 targets: 0.75 s of sim time),
    then gives a still arm 0.5 s; False then means stopped: the failed goal is no longer streamed."""
    r = make(clock=FakeClock(step=0.01))
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p + [0.0, 0.0, -0.1], R), wait_for_execution=True)  # the arm never moves
    assert 1.25e9 <= r.node.get_clock().t <= 1.3e9
    assert not r._queue and r._last is None


def test_a_refused_arm_goal_leaves_the_gripper_alone(make):
    """Review C2 M1: the arm goal is checked and planned before the gripper is commanded."""
    r = make()
    joints = {f"{j}.pos": v for j, v in zip(ARM, HOME)}
    assert not r.send_action({"fr3_joint1.pos": 0.0, "fr3_finger_joint1.pos": 0.04})  # partial arm goal
    assert not r.send_action({**joints, "fr3_joint1.pos": 10.0, "fr3_finger_joint1.pos": 0.04})  # beyond the limits
    assert np.isnan(r._width) and r._last is None and not r._queue
    assert r.send_action({**joints, "fr3_finger_joint1.pos": 0.04})
    assert r._width == pytest.approx(0.08) and r._queue


def test_failures_return_false_instead_of_raising(make):
    """Review C2 I1/M2, spec 8: send_action and reset_scene answer False with a log line, never raise."""
    r = make()
    p, R = r._motion.ee_pose(HOME)
    zero = pose(p, R)
    zero.orientation.x = zero.orientation.y = zero.orientation.z = zero.orientation.w = 0.0
    assert not r.send_action(zero)  # scipy: ValueError, zero quaternion
    assert not r.send_action(pose(p, R), speed=0.0)  # plan(): ValueError, speed must be > 0
    assert not r.send_action(pose(p + [np.nan, 0.0, 0.0], R))  # plan(): PlanningError, non-finite goal

    def silent(service, request, message=None):
        raise TimeoutError("Service '/Sim_0/Reset' timed out after 60.0s")

    r.callService = silent
    assert not r.reset_scene("/Sim_0/Reset", "request", HOME)
    assert [type_ for type_ in ("ValueError", "ValueError", "PlanningError", "TimeoutError")] == [
        next(t for t in ("ValueError", "PlanningError", "TimeoutError") if t in str(line)) for line in r.logs]


class FrameClock(FakeClock):
    """Advances like FakeClock but reads in whole frames, as /clock does with one message per rendered frame."""

    def __init__(self, step=0.004, frame=0.02):
        super().__init__(step)
        self.frame = int(frame * 1e9)

    def now(self):
        t = super().now().nanoseconds
        return Time(nanoseconds=t - t % self.frame)


def test_the_stream_keeps_20_hz_when_frames_do_not_divide_50_ms(make):
    """Review C2 M5: at step_freq 50 (20 ms frames) a period must not stretch to 60 ms."""
    from guide_motion.targets import Target

    clock = FrameClock()
    r = make(clock=clock)
    sent = []
    r._motion_target_msg = lambda target, width: clock.t
    r._motion_pub = SimpleNamespace(publish=sent.append)
    r._stop_stream = threading.Event()
    r._last = Target()
    thread = threading.Thread(target=r._stream, daemon=True)
    thread.start()
    while clock.t < 2.1e9:
        time.sleep(0.01)
    r._stop_stream.set()
    thread.join()
    assert 39 <= sum(t < 2.0e9 for t in sent) <= 41  # 20 Hz over 2 s of sim time


def test_a_stalled_clock_says_so(make):
    """Review C2 M8: a stopped simulator is not a timeout on sim time."""
    from guide_motion.targets import Target

    r = make(clock=FakeClock(step=0.0))
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p + [0.1, 0, 0], rotation=R), timeout=1.0, stall_timeout=0.05)
    assert "did not advance" in str(r.logs[-1])


def test_nothing_is_sent_when_no_oscbf_host_listens(make):
    """Final review I2: a scene without an OSCBF host (registered under moveit, or a second registration)
    would swallow every target and time out each try; say why instead."""
    r = make()
    r._motion_pub.get_subscription_count = lambda: 0
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p + [0.0, 0.0, -0.1], R), wait_for_execution=True)
    assert not r._queue and r.logs[-1] == ("no OSCBF host subscribes to /Sim_0/Scene_0/franka/motion_target: "
                                           "is GUIDE running with controller: oscbf for this scene?")


def test_a_joint_goal_waits_for_the_joints_not_only_the_tool(make):
    """Final review M2: joint 7 0.02 rad off turns the tool 1.15 deg about its own axis, inside the tool
    tolerance (5 mm, 2 deg), but the arm is not at the joint goal yet."""
    from guide_motion.targets import JOINTS, Target

    r = make()
    goal = np.array(HOME)
    goal[6] += 0.02
    assert not r._wait_reached(Target(mode=JOINTS, joints=goal), timeout=0.3)
    assert r._wait_reached(Target(mode=JOINTS, joints=np.array(HOME)), timeout=0.3)


def test_an_unknown_controller_is_refused_before_anything_starts(make):
    """Final review M3: anything but moveit or oscbf used to mean MoveIt, against a scene with no move_group."""
    r = make(controller="OSCBF")
    r.node = None  # not connected yet
    with pytest.raises(ValueError, match="controller must be 'moveit' or 'oscbf', got 'OSCBF'"):
        r.connect()
