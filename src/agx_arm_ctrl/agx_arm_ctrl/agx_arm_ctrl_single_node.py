#!/usr/bin/env python3
# -*-coding:utf8-*-
import time
import rclpy
import math
import statistics
import threading
from typing import Optional
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW, NeroFW
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool, Empty, Trigger
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from scipy.spatial.transform import Rotation as R

from agx_arm_msgs.msg import (
    AgxArmStatus, GripperStatus,
    HandStatus, HandCmd, HandPositionTimeCmd,
    MoveMITMsg
)
from agx_arm_ctrl.effector import AgxGripperWrapper, Revo2Wrapper

GRIPPER_JOINT_NAME = "gripper"

# rad; limit에 정확히 걸치는 것을 피하려면 작은 안쪽 여유값을 줄 수 있음
JOINT_LIMIT_MARGIN = 0.01

REVO2_FINGER_CONFIG = [
    # (joint_name, attribute_name, max_angle)
    ("thumb_metacarpal_joint", "thumb_base", 1.57),
    ("thumb_proximal_joint", "thumb_tip", 1.03),
    ("index_proximal_joint", "index_finger", 1.41),
    ("middle_proximal_joint", "middle_finger", 1.41),
    ("ring_proximal_joint", "ring_finger", 1.41),
    ("pinky_proximal_joint", "pinky_finger", 1.41),
]

REVO2_LEFT_HAND_JOINT_NAMES = [f"left_{suffix}" for suffix, _, _ in REVO2_FINGER_CONFIG]
REVO2_RIGHT_HAND_JOINT_NAMES = [f"right_{suffix}" for suffix, _, _ in REVO2_FINGER_CONFIG]
REVO2_HAND_JOINT_NAMES = REVO2_LEFT_HAND_JOINT_NAMES + REVO2_RIGHT_HAND_JOINT_NAMES

REVO2_HAND_JOINT_TO_FINGER_ATTR = {
    f"{prefix}{suffix}": (attr, max_angle)
    for prefix in ("left_", "right_")
    for suffix, attr, max_angle in REVO2_FINGER_CONFIG
}

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyAgxArm.api.agx_arm_factory import PiperCanDefaultConfig

class AgxArmRosNode(Node):

    def __init__(self):
        super().__init__("agx_arm_ctrl_single_node")

        ### ros parameters
        self._declare_parameters()
        self._load_parameters()
        self._log_parameters()

        ### AgxArmFactory
        self._init_agx_arm()

        ### effector
        self._init_effector()

        ### startup joint-limit clip
        if self.enforce_joint_limits_on_start:
            self.clip_joint_limit()

        ### publishers
        self._setup_publishers()

        ### subscribers
        self._setup_subscribers()

        ### services
        self._setup_services()

        ### publisher thread
        self.publisher_thread = threading.Thread(target=self._publish_thread)
        self.publisher_thread.start()

        ### collision guard: FJT cancel client + watchdog thread
        self._fjt_cancel_cli = self.create_client(
            CancelGoal, f"{self.arm_controller_action}/_action/cancel_goal"
        )
        self.collision_thread = threading.Thread(target=self._collision_watchdog, daemon=True)
        self.collision_thread.start()

    ### initialization methods
    def _declare_parameters(self):
        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "piper")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("fast_mode", False)
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 200)
        self.declare_parameter("enable_timeout", 5.0)
        self.declare_parameter("effector_type", "none")
        self.declare_parameter("tcp_offset", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("gripper_default_effort", 1.0)
        self.declare_parameter("publish_gripper_joint", True)
        self.declare_parameter("control_enabled", True)
        self.declare_parameter("enforce_joint_limits_on_start", True)
        # 충돌 자동 leader 전환 가드
        self.declare_parameter("collision_guard_enabled", False)
        # 관절별 crash protection rating 배열(길이=관절수). 각 0~8(0=해당 관절 끔), 전부 0=전체 끔.
        self.declare_parameter("crash_protection_level", [0, 0, 0, 0, 0, 0, 0])
        self.declare_parameter("collision_window_sec", 3.0)      # 정지 판정 윈도우
        self.declare_parameter("collision_std_deg", 2.0)         # 정지 판정 joint 표준편차 임계(도)
        self.declare_parameter("arm_controller_action", "arm_controller/follow_joint_trajectory")

    def _load_parameters(self):
        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.effector_type = self.get_parameter("effector_type").value
        self.tcp_offset = self.get_parameter("tcp_offset").value
        self.gripper_default_effort = self.get_parameter("gripper_default_effort").value
        self.publish_gripper_joint = self.get_parameter("publish_gripper_joint").value
        self.control_enabled = self.get_parameter("control_enabled").value
        self.enforce_joint_limits_on_start = self.get_parameter("enforce_joint_limits_on_start").value
        self.collision_guard_enabled = self.get_parameter("collision_guard_enabled").value
        self.crash_protection_level = self.get_parameter("crash_protection_level").value
        self.collision_window_sec = self.get_parameter("collision_window_sec").value
        self.collision_std_deg = self.get_parameter("collision_std_deg").value
        self.arm_controller_action = self.get_parameter("arm_controller_action").value

        if self.arm_type not in ArmModel.__dict__.values():
            self.get_logger().error(
                f"Unsupported arm_type '{self.arm_type}', expected one of {list(ArmModel.__dict__.values())}."
            )
            exit(1)

        if self.gripper_default_effort < 0:
            self.get_logger().warn(
                f"gripper_default_effort should be greater than 0, but got {self.gripper_default_effort}. "
                "Setting it to default value 1.0"
            )
            self.gripper_default_effort = 1.0

        ### variables
        self.is_piper = "piper" in self.arm_type
        self.is_nero = "nero" in self.arm_type
        self.is_switch_seamlessly = True
        self.is_mit_mode = False
        self.enable_flag = False
        self.control_ready = False
        self._control_ready_logged = False
        self.arm_joint_names = list()
        self.arm_joint_limits = list()
        self.arm_joint_count = 0
        self._control_gate_block_logged = False
        self.drag_mode_active = False         # 끌리는 모드(leader zero-force drag) 활성 여부
        self._drag_exit_use_follower = False  # 해제 시 follower 모드 사용(nero >= 1.12)
        self._drag_mode_block_logged = False  # 드래그 중 제어 차단 로그 1회용
        # 충돌 자동 leader 전환 가드 상태
        self.collision_guard_on = self.collision_guard_enabled  # 가드 on/off (collision_guard 서비스로 토글)
        self.collision_active = False         # 충돌→leader 전환 중(이 동안 control/* 차단)
        self._collision_block_logged = False  # 충돌 중 제어 차단 로그 1회용

    def _log_parameters(self):
        self.get_logger().info(f"can_port: {self.can_port}")
        self.get_logger().info(f"arm_type: {self.arm_type}")
        self.get_logger().info(f"auto_enable: {self.auto_enable}")
        self.get_logger().info(f"fast_mode: {self.fast_mode}")
        self.get_logger().info(f"speed_percent: {self.speed_percent}")
        self.get_logger().info(f"pub_rate: {self.pub_rate}")
        self.get_logger().info(f"enable_timeout: {self.enable_timeout}")
        self.get_logger().info(f"effector_type: {self.effector_type}")
        self.get_logger().info(f"tcp_offset: {self.tcp_offset}")
        self.get_logger().info(f"gripper_default_effort: {self.gripper_default_effort}")
        self.get_logger().info(f"publish_gripper_joint: {self.publish_gripper_joint}")
        self.get_logger().info(f"control_enabled: {self.control_enabled}")
        self.get_logger().info(f"enforce_joint_limits_on_start: {self.enforce_joint_limits_on_start}")
        self.get_logger().info(f"collision_guard_enabled: {self.collision_guard_enabled}")
        self.get_logger().info(f"crash_protection_level: {self.crash_protection_level}")
        self.get_logger().info(f"collision_window_sec: {self.collision_window_sec}")
        self.get_logger().info(f"collision_std_deg: {self.collision_std_deg}")

    def _init_agx_arm(self):
        config: PiperCanDefaultConfig = create_agx_arm_config(
            robot=self.arm_type, comm="can", channel=self.can_port
        )
        self.agx_arm = AgxArmFactory.create_arm(config)
        self.agx_arm.connect()

        self.arm_joint_names = list(config["joint_limits"].keys())
        self.arm_joint_limits = list(config["joint_limits"].values())  # 관절 순서의 [[min, max], ...]
        self.arm_joint_count = self.agx_arm.joint_nums

        # 펌웨어 버전 확정 + 필요 시 해당 드라이버로 재연결을 enable 전에 수행한다.
        # (crash protection 레벨은 disable 상태 + 드라이버 확정 후에만 변경 가능 — 실측)
        start_time = time.time()
        self.firmware = None
        while time.time() - start_time < self.enable_timeout:
            self.firmware = self.agx_arm.get_firmware()
            if self.firmware:
                break
            time.sleep(0.005)

        if self.firmware:
            current_version = self.firmware['software_version']
            self.get_logger().info(f"firmware version: {current_version}")
            firmeware_version = PiperFW.DEFAULT
            if self.is_piper:
                if current_version < "S-V1.8-5":
                    self.is_switch_seamlessly = False
                if current_version > "S-V1.8-2" and current_version < "S-V1.8-8":
                    firmeware_version = PiperFW.V183
                elif current_version >= "S-V1.8-8":
                    firmeware_version = PiperFW.V188
            elif self.is_nero:
                if current_version == "1.11":
                    firmeware_version = NeroFW.V111
                elif current_version >= "1.12":
                    firmeware_version = NeroFW.V112
                # nero >= 1.12는 leader 모드 해제 시 set_follower_mode 사용
                # (set_normal_mode은 no-op). piper/nero <= 1.11은 set_normal_mode.
                self._drag_exit_use_follower = current_version >= "1.12"

            if firmeware_version != PiperFW.DEFAULT:
                self.agx_arm.disconnect()
                config = create_agx_arm_config(
                    robot=self.arm_type, comm="can", channel=self.can_port,
                    firmeware_version=firmeware_version
                )
                self.agx_arm = AgxArmFactory.create_arm(config)
                self.agx_arm.connect()
        else:
            self.get_logger().error("Failed to get firmware version")
            exit(1)

        # crash protection 레벨 — 관절별 정수 배열(길이=관절수, joint_index 1~7에 1:1). 각 0~8(0=해당 관절 끔).
        # 전부 0(또는 빈 배열)이면 전체 끔으로 보고 disable/set 생략(limp 없음).
        # 드라이버 확정 후 + 모터 disable 상태에서만 변경 가능(실측). enable 상태에선 펌웨어가
        # 레벨 변경을 거부하므로(-> False), set 전에 명시적으로 disable한다. (disconnect/connect는 CAN
        # 소켓 재오픈일 뿐 모터 파워사이클이 아니라, 핫 리런치 시 팔이 직전 enable 상태를 유지 →
        # set이 거부됨. 따라서 여기서 강제로 disable→set(관절별)→enable.)
        levels = [int(v) for v in (self.crash_protection_level or [])]
        if any(v > 0 for v in levels):
            if len(levels) != self.arm_joint_count:
                self.get_logger().error(
                    f"crash_protection_level length {len(levels)} != joint count "
                    f"{self.arm_joint_count}; skipping crash protection setup"
                )
            else:
                try:
                    self._enable_arm(False, self.enable_timeout)   # disable (retry+verify)
                    time.sleep(0.6)                                 # 레벨 변경 수락 대기(실측 타이밍)
                    results = []
                    for i, rating in enumerate(levels):
                        ok = self.agx_arm.set_crash_protection_rating(
                            joint_index=i + 1, rating=rating        # joint_index는 1-indexed
                        )
                        results.append(ok)
                        time.sleep(0.05)                            # CAN 과도송신 방지
                    self.get_logger().info(
                        f"set_crash_protection_rating per-joint {levels} -> {results}"
                    )
                    time.sleep(0.3)
                except Exception as e:
                    self.get_logger().warn(f"set_crash_protection_rating failed: {e}")

        # enable (crash 설정 이후)
        if self.auto_enable:
            if not self._enable_arm(True, self.enable_timeout):
                self.get_logger().error("Failed to auto-enable the arm")
        else:
            time.sleep(0.1)
            self.enable_flag = self.agx_arm.get_joint_enable_status(255)

        self.agx_arm.set_speed_percent(self.speed_percent)
        self.agx_arm.set_tcp_offset(self.tcp_offset)

    def _init_effector(self):
        self.gripper: Optional[AgxGripperWrapper] = None
        self.hand: Optional[Revo2Wrapper] = None

        if self.effector_type == "agx_gripper":
            self.gripper = AgxGripperWrapper(self.agx_arm)
            if self.gripper.initialize():
                self.get_logger().info("AgxGripper initialized successfully")
            else:
                self.get_logger().error("Failed to initialize AgxGripper")
                self.gripper = None
        elif self.effector_type == "revo2" and self.is_switch_seamlessly:
            self.hand = Revo2Wrapper(self.agx_arm)
            if self.hand.initialize():
                self.get_logger().info("Revo2 hand initialized successfully")
            else:
                self.get_logger().error("Failed to initialize Revo2 hand")
                self.hand = None

    def _setup_publishers(self):
        self.joint_states_pub = self.create_publisher(
            JointState, "feedback/joint_states", 1
        )
        # self.flange_pose_pub = self.create_publisher(
        #     PoseStamped, "feedback/flange_pose", 1
        # )
        self.tcp_pose_pub = self.create_publisher(
            PoseStamped, "feedback/tcp_pose", 1
        )
        self.arm_status_pub = self.create_publisher(
            AgxArmStatus, "feedback/arm_status", 1
        )
        self.leader_joint_states_pub = self.create_publisher(
            JointState, "feedback/leader_joint_states", 1
        )
        if self.gripper is not None:
            self.gripper_status_pub = self.create_publisher(
                GripperStatus, "feedback/gripper_status", 1
            )
        if self.hand is not None:
            self.hand_status_pub = self.create_publisher(
                HandStatus, "feedback/hand_status", 1
            )

    def _setup_subscribers(self):
        self.create_subscription(
            JointState, "control/joint_states", self._joint_states_callback, 1
        )
        self.create_subscription(
            JointState, "control/move_j", self._move_j_callback, 1
        )
        self.create_subscription(
            PoseStamped, "control/move_p", self._move_p_callback, 1
        )
        self.create_subscription(
            PoseStamped, "control/move_l", self._move_l_callback, 1
        )
        self.create_subscription(
            PoseArray, "control/move_c", self._move_c_callback, 1
        )
        self.create_subscription(
            JointState, "control/move_js", self._move_js_callback, 1
        )
        self.create_subscription(
            MoveMITMsg, "control/move_mit", self._move_mit_callback, 1
        )
        if self.hand is not None:
            self.create_subscription(
                HandCmd, "control/hand", self._hand_cmd_callback, 1
            )
            self.create_subscription(
                HandPositionTimeCmd, "control/hand_position_time", 
                self._hand_position_time_cmd_callback, 1
            )

    def _setup_services(self):
        self.create_service(SetBool, "enable_agx_arm", self._enable_callback)
        self.create_service(SetBool, "control_enable", self._control_gate_callback)
        self.create_service(SetBool, "drag_mode", self._drag_mode_callback)
        self.create_service(SetBool, "collision_guard", self._collision_guard_callback)
        self.create_service(Trigger, "get_crash_protection_rating", self._get_crash_rating_callback)
        self.create_service(Empty, "move_home", self._move_home_callback)
        self.create_service(Empty, "emergency_stop", self._emergency_stop_callback)
        if not self.is_switch_seamlessly:
            self.create_service(Empty, "exit_teach_mode", self._exit_teach_mode_callback)

    ### utility methods
    def _float_to_ros_time(self, timestamp: float) -> Time:
        """Convert float timestamp to ROS Time message """
        ros_time = Time()
        ros_time.sec = int(timestamp)
        ros_time.nanosec = int((timestamp - ros_time.sec) * 1e9)
        return ros_time

    def _safe_get_value(self, array, index, default=0.0) -> float:
        if index >= len(array):
            return default
        value = array[index]
        return default if math.isnan(value) else value

    def _check_arm_ready(self) -> bool:
        joint_states = self.agx_arm.get_joint_angles()
        if joint_states is None or joint_states.hz <= 0:
            return False
        return True

    def _check_can_control(self) -> bool:
        if not self.control_ready:
            # Startup warm-up: ignore incoming control commands until a valid
            # joint state stream is available.
            return False
        if self.collision_active:
            # 충돌 감지 후 leader 전환/복구 중에는 외부 제어 명령을 무시한다.
            if not self._collision_block_logged:
                self.get_logger().info("Collision guard active (leader), ignore control commands")
                self._collision_block_logged = True
            return False
        if self.drag_mode_active:
            # 사람이 손으로 끄는 동안에는 제어 명령을 무시한다.
            if not self._drag_mode_block_logged:
                self.get_logger().info("Drag mode active, ignore control commands")
                self._drag_mode_block_logged = True
            return False
        if not self._check_arm_ready():
            self.get_logger().warn("Agx_arm is not connected, cannot control")
            return False
        if not self.enable_flag:
            self.get_logger().warn("Agx_arm is not enabled, cannot control")
            return False
        if not self.control_enabled:
            if not self._control_gate_block_logged:
                self.get_logger().info(
                    "External control gate is closed, ignore control commands"
                )
                self._control_gate_block_logged = True
            return False
        self._control_gate_block_logged = False
        if not self.is_switch_seamlessly:
            arm_status = self.agx_arm.get_arm_status()
            if arm_status is not None and arm_status.msg.ctrl_mode == self.agx_arm.ARM_STATUS.CtrlMode.TEACHING_MODE:
                self.get_logger().warn("Agx_arm is in teach mode, cannot control")
                return False
        return True

    def _create_pose_cmd(self, pose: Pose) -> list:
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        pose_xyz = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]
        euler_angles = R.from_quat(quaternion).as_euler("xyz", degrees=False)
        tcp_pose = pose_xyz + euler_angles.tolist()
        flange_pose = self.agx_arm.get_tcp2flange_pose(tcp_pose)
        return flange_pose

    def _wait_motion_done(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        start_time = time.time()

        while True:
            status = self.agx_arm.get_arm_status()
            if status is not None and status.msg.motion_status == self.agx_arm.ARM_STATUS.MotionStatus.REACH_TARGET_POS_SUCCESSFULLY:
                return True
            
            if time.time() - start_time > timeout:
                self.get_logger().error(
                    f"Timeout waiting for arm to motion done after {timeout} seconds"
                )
                return False
            time.sleep(poll_interval)

    def _enable_arm(self, enable: bool = True, timeout: float = 5.0) -> bool:
        start_time = time.time()
        action_name = "enable" if enable else "disable"
        
        while not (self.agx_arm.enable() if enable else self.agx_arm.disable()):
            if time.time() - start_time > timeout:
                self.get_logger().error(
                    f"Timeout waiting for arm to {action_name} after {timeout} seconds"
                )
                return False
            time.sleep(1)
        
        joints_status = self.agx_arm.get_joint_enable_status(255)
        all_joints_in_target_status = joints_status if enable else not joints_status

        if all_joints_in_target_status:
            self.enable_flag = True if enable else False
            self.get_logger().info(f"All joints {action_name} status is {self.enable_flag}")
        else:
            self.get_logger().warn(
                f"Not all joints are {action_name}d after {action_name}ing the arm"
            )
        
        return True

    def _set_normal_control_mode(self) -> None:
        """leader(드래그) 모드에서 정상 CAN 제어 모드로 복귀한다.

        펌웨어 >= 1.12(nero)는 set_follower_mode, 그 외(piper/nero <= 1.11)는
        set_normal_mode을 사용한다.
        """
        if self._drag_exit_use_follower:
            self.agx_arm.set_follower_mode()
        else:
            self.agx_arm.set_normal_mode()

    def clip_joint_limit(self) -> None:
        """현재 관절 각도를 읽어, joint limit을 벗어난 관절을 limit 안쪽으로 move_j로 보정한다.

        자기완결형 메소드로, 피드백/enable/teach 모드를 직접 확인한 뒤 동작한다.
        """
        # 관절 피드백이 들어올 때까지 제한 대기
        start_time = time.time()
        while not self._check_arm_ready():
            if time.time() - start_time > self.enable_timeout:
                self.get_logger().warn(
                    "No joint feedback available, skip startup joint-limit clip"
                )
                return
            # time.sleep(0.01)

        if not self.enable_flag:
            self.get_logger().warn(
                "Agx_arm is not enabled, joint-limit clip command will not apply, skip"
            )
            return

        if not self.is_switch_seamlessly:
            arm_status = self.agx_arm.get_arm_status()
            if arm_status is not None and arm_status.msg.ctrl_mode == self.agx_arm.ARM_STATUS.CtrlMode.TEACHING_MODE:
                self.get_logger().warn(
                    "Agx_arm is in teach mode, skip startup joint-limit clip"
                )
                return

        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            self.get_logger().warn("No valid joint angles, skip startup joint-limit clip")
            return

        current = list(js.msg)
        target = list(current)
        violations = []
        for idx, (name, (lo, hi)) in enumerate(zip(self.arm_joint_names, self.arm_joint_limits)):
            if idx >= len(current):
                break
            clamped = min(max(current[idx], lo + JOINT_LIMIT_MARGIN), hi - JOINT_LIMIT_MARGIN)
            if clamped != current[idx]:
                violations.append((name, current[idx], clamped, lo, hi))
                target[idx] = clamped

        if not violations:
            self.get_logger().info("All joints are within limits at startup")
            return

        for name, cur, clamped, lo, hi in violations:
            self.get_logger().warn(
                f"Joint '{name}' out of limit: {cur:.4f} rad not in [{lo:.4f}, {hi:.4f}], "
                f"moving to {clamped:.4f} rad"
            )

        self.agx_arm.move_j(target)
        self.is_mit_mode = False
        if self._wait_motion_done():
            self.get_logger().info("Startup joint-limit clip completed")

    ### publisher thread
    def _publish_thread(self):
        rate = self.create_rate(self.pub_rate)

        # publishing loop
        while rclpy.ok():
            if self.agx_arm.is_ok():
                if not self.control_ready and self._check_arm_ready():
                    self.control_ready = True
                    if not self._control_ready_logged:
                        self.get_logger().info("Agx_arm feedback is ready, control is now enabled")
                        self._control_ready_logged = True
                self._publish_joint_states()
                self._publish_pose()
                self._publish_arm_status()
                self._publish_effector_status()
                self._publish_leader_joint_states()
            rate.sleep()
    
    ### publish methods
    def _get_gripper_joint_data(self):
        if self.gripper is None or not self.gripper.is_ok():
            return []
        status = self.gripper.get_status()
        if status is None:
            return []

        gripper_joint_map = {
            "gripper_joint1":     0.5,
            "gripper_joint2":    -0.5,
        }
        if self.publish_gripper_joint:
            gripper_joint_map[GRIPPER_JOINT_NAME] = 1.0

        return [
            (name, status.width * scale, 0.0, status.force)
            for name, scale in gripper_joint_map.items()
        ]

    def _get_gripper_joint_ctrl_data(self):
        if self.gripper is None:
            return []
        ctrl_states = self.gripper.get_ctrl_states()
        if ctrl_states is None:
            return []

        gripper_joint_map = {
            "gripper_joint1":     0.5,
            "gripper_joint2":    -0.5,
        }
        if self.publish_gripper_joint:
            gripper_joint_map[GRIPPER_JOINT_NAME] = 1.0

        return [
            (name, ctrl_states.width * scale, 0.0, ctrl_states.force)
            for name, scale in gripper_joint_map.items()
        ]

    def _get_hand_joint_data(self):
        if self.hand is None or not self.hand.is_ok():
            return []
        finger_pos = self.hand.get_finger_position()
        if finger_pos is None:
            return []
        joint_names = REVO2_LEFT_HAND_JOINT_NAMES if self.hand.is_hand_left() else REVO2_RIGHT_HAND_JOINT_NAMES
        
        result = []
        for joint_name in joint_names:
            attr = REVO2_HAND_JOINT_TO_FINGER_ATTR[joint_name][0]
            max_angle = REVO2_HAND_JOINT_TO_FINGER_ATTR[joint_name][1]
            joint_value = max(0.0, min(max_angle, getattr(finger_pos, attr, 0) * max_angle / 100))
            result.append((joint_name, joint_value, 0.0, 0.0))
        
        return result

    def _publish_joint_states(self):
        joint_states = self.agx_arm.get_joint_angles()
        if joint_states is None or joint_states.hz <= 0:
            return

        velocitys = []
        efforts = []
        for joint_index in range(1, self.arm_joint_count+1):
            ms = self.agx_arm.get_motor_states(joint_index)
            if ms is None:
                return
            velocitys.append(ms.msg.velocity)
            efforts.append(ms.msg.torque)

        msg = JointState()
        msg.header.stamp = self._float_to_ros_time(joint_states.timestamp)
        
        joints_data = []
        # arm
        joints_data.extend(
            (joint_name, joint_state, velocity, effort)
            for joint_name, joint_state, velocity, effort in zip(self.arm_joint_names, joint_states.msg, velocitys, efforts)
        )
        # gripper
        joints_data.extend(self._get_gripper_joint_data())
        # hand
        joints_data.extend(self._get_hand_joint_data())
        if joints_data:
            msg.name, msg.position, msg.velocity, msg.effort =map(list, zip(*joints_data))
            self.joint_states_pub.publish(msg)

    def _publish_pose(self):
        flange_pose = self.agx_arm.get_flange_pose()
        if flange_pose is None or flange_pose.hz <= 0:
            return
        
        tcp_pose = self.agx_arm.get_flange2tcp_pose(flange_pose.msg)

        # pose1 = Pose()
        # pose1.position.x, pose1.position.y, pose1.position.z = flange_pose.msg[0:3]
        # roll, pitch, yaw = flange_pose.msg[3:6]
        # quaternion = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        # pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w = quaternion

        pose2 = Pose()
        pose2.position.x, pose2.position.y, pose2.position.z = tcp_pose[0:3]
        roll, pitch, yaw = tcp_pose[3:6]
        quaternion = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        pose2.orientation.x, pose2.orientation.y, pose2.orientation.z, pose2.orientation.w = quaternion

        msg = PoseStamped()
        msg.header.stamp = self._float_to_ros_time(flange_pose.timestamp)
        # msg.pose = pose1
        # self.flange_pose_pub.publish(msg)
        msg.pose = pose2
        self.tcp_pose_pub.publish(msg)

    def _publish_arm_status(self):
        arm_status = self.agx_arm.get_arm_status()
        if arm_status is None:
            return

        msg = AgxArmStatus()
        msg.ctrl_mode = arm_status.msg.ctrl_mode
        msg.arm_status = arm_status.msg.arm_status
        msg.mode_feedback = arm_status.msg.mode_feedback
        msg.teach_status = arm_status.msg.teach_status
        msg.motion_status = arm_status.msg.motion_status
        msg.trajectory_num = arm_status.msg.trajectory_num
        err = arm_status.msg.err_status
        for i in range(self.arm_joint_count):
            angle_limit = getattr(err, f"joint_{i+1}_angle_limit")
            comm_status = getattr(err, f"communication_status_joint_{i+1}")

            msg.joint_angle_limit.append(angle_limit)
            msg.communication_status_joint.append(comm_status)

        self.arm_status_pub.publish(msg)

    def _publish_leader_joint_states(self):
        leader_joint_angles = self.agx_arm.get_leader_joint_angles()
        if leader_joint_angles is None:
            return

        msg = JointState()
        msg.header.stamp = self._float_to_ros_time(leader_joint_angles.timestamp)
        names = list(self.arm_joint_names)
        positions = list(leader_joint_angles.msg)
        velocity = [0.0] * self.arm_joint_count
        effort = [0.0] * self.arm_joint_count

        for name, pos, vel, eff in self._get_gripper_joint_ctrl_data():
            names.append(name)
            positions.append(pos)
            velocity.append(vel)
            effort.append(eff)

        msg.name = names
        msg.position = positions
        msg.velocity = velocity
        msg.effort = effort
        self.leader_joint_states_pub.publish(msg)

    def _publish_gripper_status(self):
        status = self.gripper.get_status()
        if status is not None:
            msg = GripperStatus()
            msg.header.stamp = self._float_to_ros_time(status.timestamp)
            msg.width = status.width
            msg.force = status.force
            msg.voltage_too_low = status.voltage_too_low
            msg.motor_overheating = status.motor_overheating
            msg.driver_overcurrent = status.driver_overcurrent
            msg.driver_overheating = status.driver_overheating
            msg.sensor_status = status.sensor_status
            msg.driver_error_status = status.driver_error_status
            msg.driver_enable_status = status.driver_enable_status
            msg.homing_status = status.homing_status
            self.gripper_status_pub.publish(msg)

    def _publish_hand_status(self):
        hand_status = self.hand.get_status()
        finger_pos = self.hand.get_finger_position()
        if hand_status is not None:
            msg = HandStatus()
            msg.header.stamp = self._float_to_ros_time(hand_status.timestamp)
            msg.left_or_right = hand_status.left_or_right
            # status
            msg.thumb_tip_status = hand_status.thumb_tip
            msg.thumb_base_status = hand_status.thumb_base
            msg.index_finger_status = hand_status.index_finger
            msg.middle_finger_status = hand_status.middle_finger
            msg.ring_finger_status = hand_status.ring_finger
            msg.pinky_finger_status = hand_status.pinky_finger
            # position
            if finger_pos is not None:
                msg.thumb_tip_pos = finger_pos.thumb_tip
                msg.thumb_base_pos = finger_pos.thumb_base
                msg.index_finger_pos = finger_pos.index_finger
                msg.middle_finger_pos = finger_pos.middle_finger
                msg.ring_finger_pos = finger_pos.ring_finger
                msg.pinky_finger_pos = finger_pos.pinky_finger
            self.hand_status_pub.publish(msg)

    def _publish_effector_status(self):
        if self.gripper is not None and self.gripper.is_ok():
            self._publish_gripper_status()
        if self.hand is not None and self.hand.is_ok():
            self._publish_hand_status()

    ### arm control callbacks
    def _control_arm_joints(self, joint_pos):
        arm_joints = {
            name : value
            for name, value in joint_pos.items()
            if name in self.arm_joint_names
        }
        if arm_joints:
            joints = [arm_joints.get(name, 0) for name in self.arm_joint_names]
            if self.fast_mode:
                self.agx_arm.move_js(joints)
                self.is_mit_mode = True
            else:
                self.agx_arm.move_j(joints)
                self.is_mit_mode = False

    def _control_gripper_joint(self, joint_pos, joint_effort):
        if self.gripper is None:
            return

        # gripper_name → width scale
        gripper_joint_map = {
            GRIPPER_JOINT_NAME:   1.0,
            "gripper_joint1":    2.0,
            "gripper_joint2":    2.0,
        }

        matched = next(
            ((name, scale) for name, scale in gripper_joint_map.items()
             if name in joint_pos),
            None,
        )
        if matched is None:
            return

        joint_name, scale = matched
        width = abs(joint_pos[joint_name]) * scale
        # Use default force if effort is 0 or not specified
        force = joint_effort.get(joint_name, self.gripper_default_effort) or self.gripper_default_effort

        try:
            self.gripper.move(width=width, force=force)
        except ValueError as e:
            self.get_logger().warn(str(e))

    def _control_hand_joints(self, joint_pos):
        hand_joints = {
            name : max(0, min(100, int(value / REVO2_HAND_JOINT_TO_FINGER_ATTR[name][1] * 100)))
            for name, value in joint_pos.items()
            if name in REVO2_HAND_JOINT_NAMES
        }
        if not hand_joints:
            return
    
        if self.hand is None:
            self.get_logger().warn("revo2 hand not initialized")
            return
        finger_kwargs = {
            REVO2_HAND_JOINT_TO_FINGER_ATTR[name][0] : value
            for name, value in hand_joints.items()
            if name in REVO2_HAND_JOINT_TO_FINGER_ATTR
        }
        if finger_kwargs:
            try:
                self.hand.position_ctrl(**finger_kwargs)
            except ValueError as e:
                self.get_logger().warn(str(e))

    def _joint_states_callback(self, msg: JointState):
        if not self._check_can_control():
            return

        joint_pos = {
            name: self._safe_get_value(msg.position, idx)
            for idx, name in enumerate(msg.name)
        }
        joint_effort = {
            name: self._safe_get_value(msg.effort, idx)
            for idx, name in enumerate(msg.name)
        }
        self._control_arm_joints(joint_pos)
        self._control_gripper_joint(joint_pos, joint_effort)
        self._control_hand_joints(joint_pos)

    def _move_j_callback(self, msg: JointState):
        if not self._check_can_control():
            return

        joint_pos = {}
        for idx, joint_name in enumerate(msg.name):
            joint_pos[joint_name] = self._safe_get_value(msg.position, idx)
        joints = [joint_pos.get(i, 0) for i in self.arm_joint_names]
        self.agx_arm.move_j(joints)
        self.is_mit_mode = False

    def _move_p_callback(self, msg: PoseStamped):
        if not self._check_can_control():
            return

        pose_cmd = self._create_pose_cmd(msg.pose)
        self.agx_arm.move_p(pose_cmd)
        self.is_mit_mode = False

    def _move_l_callback(self, msg: PoseStamped):
        if not self._check_can_control():
            return

        pose_cmd = self._create_pose_cmd(msg.pose)
        self.agx_arm.move_l(pose_cmd)
        self.is_mit_mode = False

    def _move_c_callback(self, msg: PoseArray):
        if not self._check_can_control():
            return
        if len(msg.poses) < 3:
            self.get_logger().error(
                f"move_c requires at least 3 poses, but got {len(msg.poses)}"
            )
            return

        pose_start = self._create_pose_cmd(msg.poses[0])
        pose_mid = self._create_pose_cmd(msg.poses[1])
        pose_end = self._create_pose_cmd(msg.poses[2])
        self.agx_arm.move_c(pose_start, pose_mid, pose_end)
        self.is_mit_mode = False

    def _move_js_callback(self, msg: JointState):
        if not self._check_can_control():
            return

        joint_pos = {}
        for idx, joint_name in enumerate(msg.name):
            joint_pos[joint_name] = self._safe_get_value(msg.position, idx)
        joints = [joint_pos.get(i, 0) for i in self.arm_joint_names]
        self.agx_arm.move_js(joints)
        self.is_mit_mode = True

    def _move_mit_callback(self, msg: MoveMITMsg):
        if not self._check_can_control():
            return
        
        arrays = [msg.joint_index, msg.p_des, msg.v_des, msg.kp, msg.kd, msg.torque]
        if len(set(len(arr) for arr in arrays)) > 1:
            self.get_logger().error("MoveMITMsg arrays have inconsistent lengths")
            return
        
        if not arrays[0]:
            self.get_logger().warn("Received empty MoveMITMsg")
            return
        
        for i in range(len(msg.joint_index)):
            params = {
                "joint_index": msg.joint_index[i],
                "p_des": msg.p_des[i],
                "v_des": msg.v_des[i],
                "kp": msg.kp[i],
                "kd": msg.kd[i],
                "t_ff": msg.torque[i],
            }
            
            self.agx_arm.move_mit(**params)
        self.is_mit_mode = True

    ### effector control callbacks
    def _hand_position_time_cmd_callback(self, msg: HandPositionTimeCmd):
        if self.hand is None:
            self.get_logger().warn("revo2 hand not initialized")
            return
        
        try:
            self.hand.position_time_ctrl(
                mode="pos",
                thumb_tip=msg.thumb_tip_pos,
                thumb_base=msg.thumb_base_pos,
                index_finger=msg.index_finger_pos,
                middle_finger=msg.middle_finger_pos,
                ring_finger=msg.ring_finger_pos,
                pinky_finger=msg.pinky_finger_pos,
            )
            self.hand.position_time_ctrl(
                mode="time",
                thumb_tip=msg.thumb_tip_time,
                thumb_base=msg.thumb_base_time,
                index_finger=msg.index_finger_time,
                middle_finger=msg.middle_finger_time,
                ring_finger=msg.ring_finger_time,
                pinky_finger=msg.pinky_finger_time,
            )
        except ValueError as e:
            self.get_logger().error(f"hand control param error: {e}")

    def _hand_cmd_callback(self, msg: HandCmd):
        if self.hand is None:
            self.get_logger().warn("revo2 hand not initialized")
            return
        
        mode_to_method = {
            "position": self.hand.position_ctrl,
            "speed": self.hand.speed_ctrl,
            "current": self.hand.current_ctrl,
        }

        mode = msg.mode.lower()        
        if mode not in mode_to_method:
            self.get_logger().warn(f"unknown hand control mode: {mode}")
            return

        try:
            mode_to_method[mode](
                thumb_tip=msg.thumb_tip,
                thumb_base=msg.thumb_base,
                index_finger=msg.index_finger,
                middle_finger=msg.middle_finger,
                ring_finger=msg.ring_finger,
                pinky_finger=msg.pinky_finger,
            )
        except ValueError as e:
            self.get_logger().error(f"hand control param error: {e}")

    ### service callbacks
    def _enable_callback(self, request, response):
        try:
            if not self._check_arm_ready():
                response.success = False
                response.message = "Agx_arm is not connected"
                self.get_logger().warn("Agx_arm is not connected, cannot set enable state")
            elif request.data:
                response.success = True if self._enable_arm(True) else False
                response.message = "Agx_arm enabled" if response.success else "Failed to enable Agx_arm"
            else:
                response.success = True if self._enable_arm(False) else False
                response.message = "Agx_arm disabled" if response.success else "Failed to disable Agx_arm"
            
        except Exception as e:
            response.success = False
            response.message = f"Exception occurred: {str(e)}"
            self.get_logger().error(f"Failed to set enable state: {str(e)}")
        return response

    def _move_home_callback(self, request, response):
        try:
            if not self._check_arm_ready():
                self.get_logger().warn("Agx_arm is not connected, cannot move to home position")
            elif not self.enable_flag:
                self.get_logger().warn("Agx_arm is not enabled, cannot move to home position")
            elif self.drag_mode_active:
                self.get_logger().warn("Drag mode is active, disable it before move to home position")
            else:
                if not self.is_switch_seamlessly:
                    arm_status = self.agx_arm.get_arm_status()
                    if arm_status is not None and arm_status.msg.ctrl_mode == self.agx_arm.ARM_STATUS.CtrlMode.TEACHING_MODE:
                        self.get_logger().warn("Agx_arm is in teach mode, cannot move to home position")
                        return response
                    
                if self.is_mit_mode:
                    self.agx_arm.move_js([0] * self.arm_joint_count)
                else:
                    self.agx_arm.move_j([0] * self.arm_joint_count)
                if self._wait_motion_done():
                    self.get_logger().info("Agx_arm moved to home position successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to move to home position: {str(e)}")
        return response

    def _control_gate_callback(self, request, response):
        self.control_enabled = request.data
        self._control_gate_block_logged = False
        state = "opened" if request.data else "closed"
        response.success = True
        response.message = f"External control gate {state}"
        self.get_logger().info(response.message)
        return response

    def _drag_mode_callback(self, request, response):
        """힘 빼고 사람이 끄는 대로 끌리는 모드(leader zero-force drag) ON/OFF.

        ON  : set_leader_mode()로 진입. 팔은 중력보상되어 손으로 끌 수 있다.
        OFF : 정상 제어 모드로 복귀하고, 사람이 놓아둔 현재 자세를 move_j로 hold한다.
        """
        try:
            # drag(leader) 모드에서는 일반 joint feedback push가 꺼져 _check_arm_ready()가
            # False가 되므로, 연결성은 is_ok()(leader 프레임 포함 수신 여부)로 확인한다.
            # 이렇게 해야 drag 모드에서 OFF로 다시 빠져나올 수 있다.
            if not self.agx_arm.is_ok():
                response.success = False
                response.message = "Agx_arm is not connected"
                self.get_logger().warn("Agx_arm is not connected, cannot change drag mode")
                return response
            if not self.enable_flag:
                response.success = False
                response.message = "Agx_arm is not enabled, enable it before drag mode"
                self.get_logger().warn(response.message)
                return response

            if request.data:
                self.agx_arm.set_leader_mode()
                self.drag_mode_active = True
                self.is_mit_mode = False
                response.success = True
                response.message = "Drag mode enabled (arm is free to be hand-guided)"
                self.get_logger().info(response.message)
            else:
                # 끌린 자세 먼저 읽기(leader 모드에선 leader 프레임이 유효)
                pose = None
                lja = self.agx_arm.get_leader_joint_angles()
                if lja is not None:
                    pose = list(lja.msg)

                self._set_normal_control_mode()
                self.drag_mode_active = False
                self._drag_mode_block_logged = False

                # leader 각도를 못 읽었으면 push 재개를 기다렸다가 일반 피드백으로 폴백
                if pose is None:
                    time.sleep(0.2)
                    js = self.agx_arm.get_joint_angles()
                    if js is not None and js.hz > 0:
                        pose = list(js.msg)

                if pose is not None:
                    self.agx_arm.move_j(pose)
                    self.is_mit_mode = False
                else:
                    self.get_logger().warn("Drag mode off: no valid joint angles to hold")

                response.success = True
                response.message = "Drag mode disabled (normal control restored)"
                self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Exception occurred: {str(e)}"
            self.get_logger().error(f"Failed to change drag mode: {str(e)}")
        return response

    ### collision guard (충돌 시 leader 전환 → 정지 시 normal 복귀)
    def _collision_guard_callback(self, request, response):
        """충돌 자동 leader 전환 가드 ON/OFF."""
        self.collision_guard_on = bool(request.data)
        if not self.collision_guard_on:
            self._collision_block_logged = False
        response.success = True
        response.message = f"Collision guard {'on' if self.collision_guard_on else 'off'}"
        self.get_logger().info(response.message)
        return response

    def _get_crash_rating_callback(self, request, response):
        """현재 crash protection rating(관절별 0~8)을 조회해 message로 반환한다.
        (leader/충돌 중에는 push가 꺼져 일시적으로 None일 수 있음 — normal 상태에서 조회 권장)
        """
        try:
            r = self.agx_arm.get_crash_protection_rating()
            if r is None:
                response.success = False
                response.message = "crash protection rating unavailable (None)"
            else:
                response.success = True
                response.message = str(list(r.msg))
            self.get_logger().info(f"crash protection rating: {response.message}")
        except Exception as e:
            response.success = False
            response.message = f"Exception: {e}"
            self.get_logger().warn(f"get_crash_protection_rating failed: {e}")
        return response

    def _cancel_trajectory(self) -> None:
        """진행 중인 FollowJointTrajectory goal(들)을 취소한다.

        빈 goal_info(goal_id all-zero, stamp 0) = 모든 active goal 취소. RViz/MoveIt이 보낸
        goal이라도 action server(JTC)에 cancel 요청으로 취소할 수 있어, 충돌로 중단된 긴
        궤적의 잔여 waypoint가 더 이상 흘러오지 않는다.
        """
        try:
            if not self._fjt_cancel_cli.service_is_ready():
                self.get_logger().warn("FJT cancel service not available, skip trajectory cancel")
                return
            self._fjt_cancel_cli.call_async(CancelGoal.Request())
        except Exception as e:
            self.get_logger().warn(f"FJT cancel failed: {e}")

    def _enter_leader_mode(self) -> None:
        """충돌 trip 상태에서 leader(zero-force) 진입을 verify + retry(성공까지).

        clear_joint_error → (CAN 침묵 sleep) → set_leader_mode → leader 프레임(hz>0) 확인.
        COLLISION 상태에선 set_leader_mode가 거부되므로 clear로 먼저 해소하고, clear 직후
        CAN을 잠깐 쉰 뒤 전환해야 leader가 걸린다(벤치 실측).
        """
        t_start = time.time()
        k = 0
        while rclpy.ok() and self.collision_active:
            k += 1
            self.agx_arm.clear_joint_error(255)
            time.sleep(0.3)  # clear 반영 + CAN 침묵 (없으면 set_leader_mode가 묻혀 안 걸림)
            st = self.agx_arm.get_arm_status()
            if st is not None and int(st.msg.arm_status) == 0x07:  # COLLISION_OCCURRED 잔류
                continue
            self.agx_arm.set_leader_mode()
            time.sleep(0.3)  # leader 전환 + 첫 프레임 대기
            lja = self.agx_arm.get_leader_joint_angles()
            if lja is not None and lja.hz > 0:
                self.get_logger().info(
                    f"Collision: leader entered (hz={lja.hz:.0f}, {k} tries, {time.time()-t_start:.1f}s)"
                )
                return
            self.get_logger().warn(f"Collision: leader not entered (try {k}), retry...")

    def _collision_watchdog(self) -> None:
        """충돌 가드 watchdog(별도 스레드).

        IDLE: foc collision bit 감지 → FJT 취소 + leader 전환.
        DRAG: leader 자세가 collision_window_sec 동안 모든 joint 표준편차 < collision_std_deg(도)
              로 정지하면 → normal 복귀 + 실제 자세 hold → control 자동 재개.
        """
        std_thresh = math.radians(self.collision_std_deg)
        window = self.collision_window_sec
        hist = []          # [(t, [q...]), ...] leader 자세 이력(window 슬라이딩)
        period = 1.0 / 30.0
        while rclpy.ok():
            time.sleep(period)
            if not self.collision_guard_on:
                continue
            try:
                if not self.collision_active:
                    # IDLE: 충돌 감지 (foc collision bit)
                    if not (self.control_ready and self.enable_flag and self.agx_arm.is_ok()):
                        continue
                    if self.drag_mode_active:
                        continue
                    tripped_joints = []
                    coll_bits = []
                    for j in range(1, self.arm_joint_count + 1):
                        ds = self.agx_arm.get_driver_states(j)
                        b = 1 if (ds is not None and getattr(ds.msg.foc_status, "collision_status", False)) else 0
                        coll_bits.append(b)
                        if b:
                            tripped_joints.append(j)
                    if tripped_joints:
                        # 충돌 순간: 어느 관절(들)에서 trip 됐는지 + 모든 관절 토크(전류 환산)를 함께 기록
                        torques = []
                        for j in range(1, self.arm_joint_count + 1):
                            ms = self.agx_arm.get_motor_states(j)
                            torques.append(round(ms.msg.torque, 2) if ms is not None else float("nan"))
                        self.get_logger().warn(
                            f"Collision detected: joint(s)={tripped_joints}  "
                            f"coll_bits={coll_bits}  torque(N.m)={torques} "
                            "-> cancel trajectory + enter leader"
                        )
                        self.collision_active = True   # 먼저 control/* 차단
                        self._cancel_trajectory()      # 진행 중 MoveIt 궤적 폐기(잔여 차단)
                        self._enter_leader_mode()      # leader 전환(verify+retry)
                        hist = []
                else:
                    # DRAG: leader 정지 판정 → normal 복귀
                    lja = self.agx_arm.get_leader_joint_angles()
                    now = time.time()
                    if lja is not None:
                        q = list(lja.msg)
                        hist.append((now, q))
                        hist = [(t, qq) for (t, qq) in hist if now - t <= window]
                        if len(hist) >= 5 and (now - hist[0][0]) >= window * 0.95:
                            stds = [
                                statistics.pstdev([qq[i] for (_, qq) in hist])
                                for i in range(self.arm_joint_count)
                            ]
                            if all(s < std_thresh for s in stds):
                                self.get_logger().info(
                                    f"Collision: arm settled (window {window:.0f}s, all joint "
                                    f"sigma < {self.collision_std_deg:.1f} deg) -> normal recover"
                                )
                                self._set_normal_control_mode()
                                # 복귀 후 실제 현재 자세로 hold (leader q와 미세 차이로 튀지 않도록)
                                t_n = time.time()
                                q0 = list(q)
                                while time.time() - t_n < 0.5:
                                    cq = self.agx_arm.get_joint_angles()
                                    if cq is not None and cq.hz > 0:
                                        q0 = list(cq.msg)
                                        break
                                    time.sleep(0.02)
                                self.agx_arm.move_j(q0)
                                self.is_mit_mode = False
                                self.collision_active = False
                                self._collision_block_logged = False
                                hist = []
            except Exception as e:
                self.get_logger().warn(f"collision watchdog error: {e}")

    def _emergency_stop_callback(self, request, response):
        """Emergency stop: use is_switch_seamlessly flag to decide MIT vs move_j."""
        try:
            # drag 모드에선 push가 꺼져 _check_arm_ready()가 False가 될 수 있으므로
            # 연결성은 is_ok()로 확인한다.
            if not self.agx_arm.is_ok():
                self.get_logger().warn("Agx_arm is not connected, cannot perform emergency stop")
                return response
            if not self.enable_flag:
                self.get_logger().warn("Agx_arm is not enabled, cannot perform emergency stop")
                return response

            # 드래그 모드 중이면 먼저 정상 제어 모드로 복귀해야 move_j가 적용됨
            drag_pose = None
            if self.drag_mode_active:
                lja = self.agx_arm.get_leader_joint_angles()
                if lja is not None:
                    drag_pose = list(lja.msg)
                self._set_normal_control_mode()
                self.drag_mode_active = False
                self._drag_mode_block_logged = False
                if drag_pose is None:
                    time.sleep(0.2)  # push 재개 대기 후 일반 피드백으로 폴백

            js = self.agx_arm.get_joint_angles()
            if js is not None and js.hz > 0:
                q = list(js.msg)
            elif drag_pose is not None:
                q = drag_pose
            else:
                self.get_logger().warn("No valid joint angles, cannot perform emergency stop")
                return response

            if not self.is_switch_seamlessly:
                self.agx_arm.move_js(q)
                self.is_mit_mode = True
            else:
                self.agx_arm.move_j(q)
                self.is_mit_mode = False
            self.get_logger().info(f"Emergency stop command sent to {self.arm_type}")
        except Exception as e:
            self.get_logger().error(f"Emergency stop failed: {e}")
        return response

    def _exit_teach_mode_callback(self, request, response):
        try:
            arm_status = self.agx_arm.get_arm_status()
            if not self.is_piper:
                self.get_logger().warn("exit teach mode just piper series supported")
                return response

            if arm_status is not None and arm_status.msg.ctrl_mode == self.agx_arm.ARM_STATUS.CtrlMode.TEACHING_MODE:
                self.agx_arm.move_js([0] * self.arm_joint_count)
                time.sleep(2)
                self.agx_arm.electronic_emergency_stop()
                self.agx_arm.move_j([0] * self.arm_joint_count)
                time.sleep(0.3)
                self.agx_arm.reset()
                time.sleep(0.5)
                self._enable_arm(True)
                self.agx_arm.move_j([0] * self.arm_joint_count)
                self.get_logger().info("Exited teach mode successfully")
            else:
                self.get_logger().info("Agx_arm is not in teach mode")
        except Exception as e:
            self.get_logger().error(f"Failed to exit teach mode: {e}")
        return response


def main(args=None):
    rclpy.init(args=args)

    try:
        node = AgxArmRosNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
