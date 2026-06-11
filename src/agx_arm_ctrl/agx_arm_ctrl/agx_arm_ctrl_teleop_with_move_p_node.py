#!/usr/bin/env python3
# -*-coding:utf8-*-
"""move_p 실시간 teleop 검증용 최소 노드.

키보드로 엔드(flange) pose를 6-DOF jog → 매 루프 move_p(target) 스트리밍 →
feedback/joint_states 발행.

move_p는 position-velocity(스무딩) 모드이며 "연속 실행 시 직전 목표를 덮어쓴다"고
문서화되어 있어(move_l과 달리) 연속 스트리밍이 가능하다. 이 노드는 그 추종성을 검증한다.

drag_mode(SetBool) 서비스: simple_node에서 차용한 중력보상 drag. 켜지면 move_p 스트리밍
대신 move_mit(kp=0, t_ff=G(q))로 중력보상만 걸어 수동 핸드가이드. 끄면 현재 flange pose로
target을 재시드해 move_p 스트리밍 복귀. Pinocchio + URDF 중력모델 필요(없으면 drag 비활성).

probe_z(Trigger): tool -z로 일정 스텝 전진 → |F_z| 임계 접촉 → 시작 pose로 복귀.
probe_z2(Trigger): 동일하되 스텝을 감속(decel)시켜 처음엔 큰 스텝, 목표(probe_max_distance)
근처로 갈수록 작은 스텝으로 접근. 접촉 시 조기 정지·시작 pose 복귀는 probe_z와 동일.

키보드 입력이 stdin TTY를 필요로 하므로 launch가 아니라 `ros2 run`으로 실행할 것.
"""
import sys
import math
import time
import select
import termios
import tty
import atexit
import threading
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW
from scipy.spatial.transform import Rotation as _Rot

try:
    import pinocchio as pin
    _HAS_PIN = True
except ImportError:
    _HAS_PIN = False

HALF_PI = math.pi / 2.0
PROBE_CONSEC = 3            # probe 접촉 판정 디바운스(연속 주기 수)

# 키 → (축 index 0..5, 부호). 0:x 1:y 2:z 3:roll 4:pitch 5:yaw
JOG_KEYS = {
    "w": (0, +1), "s": (0, -1),   # x
    "a": (1, +1), "d": (1, -1),   # y
    "r": (2, +1), "f": (2, -1),   # z
    "u": (3, +1), "o": (3, -1),   # roll
    "i": (4, +1), "k": (4, -1),   # pitch
    "j": (5, +1), "l": (5, -1),   # yaw
}

KEYMAP_BANNER = """\
========== move_p teleop ==========
 translation (m):  x+ w / x- s    y+ a / y- d    z+ r / z- f
 rotation (rad):   roll+ u / roll- o   pitch+ i / pitch- k   yaw+ j / yaw- l
 SPACE : 현재 실제 flange pose로 target 재동기화(re-sync)
 [ / ] : linear_step  x1/1.5 / x1.5      - / = : angular_step  x1/1.5 / x1.5
 p     : 현재 target / 실제 pose 출력
 ESC / Ctrl-C : 종료
===================================="""


# ── pose 변환 헬퍼 (T = (R 3×3, p 3)). 계산 전용, I/O 없음 ──
def _euler_to_R(rpy):
    return _Rot.from_euler("xyz", rpy).as_matrix()


def _R_to_euler(Rm):
    return _Rot.from_matrix(Rm).as_euler("xyz")


def _rotvec_to_R(v):
    return _Rot.from_rotvec(v).as_matrix()


def _compose(Ta, Tb):
    Ra, pa = Ta
    Rb, pb = Tb
    return (Ra @ Rb, pa + Ra @ pb)


def _inverse(T):
    Rm, p = T
    Rt = Rm.T
    return (Rt, -Rt @ p)


class AgxArmTeleopWithMovePNode(Node):

    def __init__(self):
        super().__init__("agx_arm_ctrl_teleop_with_move_p_node")

        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "nero")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 50)
        self.declare_parameter("enable_timeout", 5.0)
        self.declare_parameter(
            "linear_step", 0.005,
            ParameterDescriptor(
                description="키 1회당 위치 jog 증분(m)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.1, step=0.0)],
            ),
        )
        self.declare_parameter(
            "angular_step", 0.01,
            ParameterDescriptor(
                description="키 1회당 자세 jog 증분(rad)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.0)],
            ),
        )
        # drag(중력보상) 모드용 — simple_node에서 차용
        self.declare_parameter(
            "urdf_path",
            "/home/yunbeom/agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/nero_handeye.urdf",
        )
        # endeffector_force 추정 대상 EE 프레임(런타임 변경 가능)
        self.declare_parameter("ee_frame", "end_point_link")
        for j in range(1, 8):
            self.declare_parameter(
                f"gravity_scale_{j}", 1.0,
                ParameterDescriptor(
                    description=f"joint{j} 중력보상 스케일(0=무보상,1=완전)",
                    floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.0)],
                ),
            )
        for j in range(1, 8):
            self.declare_parameter(
                f"gravity_kd_{j}", 0.1,
                ParameterDescriptor(
                    description=f"joint{j} 가상 점성감쇠(drag)",
                    floating_point_range=[FloatingPointRange(from_value=0.0, to_value=5.0, step=0.0)],
                ),
            )
        # probe_z 서비스: tool -z 전진 → |F_z| 임계 접촉 → 전진거리만큼 복귀
        self.declare_parameter(
            "probe_force_threshold", 5.0,
            ParameterDescriptor(
                description="probe 접촉 판정 |F_z| 임계(N, EE프레임)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=50.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "probe_speed", 0.01,
            ParameterDescriptor(
                description="probe 전진/복귀 속도(m/s)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.1, step=0.0)],
            ),
        )
        self.declare_parameter(
            "probe_max_distance", 0.05,
            ParameterDescriptor(
                description="probe 안전 최대 전진거리(m). probe_z2의 감속 목표 거리 겸용",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.3, step=0.0)],
            ),
        )
        # probe_z2(감속 probe): 목표(probe_max_distance)에서 멀면 빠르게, 가까우면 느리게
        self.declare_parameter(
            "probe2_speed_max", 0.03,
            ParameterDescriptor(
                description="probe_z2 초기(목표에서 먼) 전진 속도(m/s) = 큰 스텝",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.2, step=0.0)],
            ),
        )
        self.declare_parameter(
            "probe2_speed_min", 0.005,
            ParameterDescriptor(
                description="probe_z2 최종(목표 근처) 전진 속도(m/s) = 작은 스텝",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.1, step=0.0)],
            ),
        )
        # force_control(힘제어 유지) 서비스: tool -z로 force_des[N]를 feedforward로 눌러 유지하고
        # 나머지 축은 위치/자세를 잡는다. move_mit 토크(kp=0, 모터측 kd 감쇠)로 송신.
        # 감쇠는 gravity_kd_j(모터측), 중력보상은 gravity_scale_j 재사용.
        self.declare_parameter(
            "force_des", 5.0,
            ParameterDescriptor(
                description="force_control 목표 누름힘(N, tool -z, feedforward)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=30.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "force_max_distance", 0.05,
            ParameterDescriptor(
                description="force_control 안전 최대 전진거리(m, 무접촉 폭주 차단)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.3, step=0.0)],
            ),
        )
        self.declare_parameter(
            "force_kp_lin", 200.0,
            ParameterDescriptor(
                description="force_control 위치유지 강성(N/m, 힘축 제외). 50Hz라 보수적으로",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2000.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "force_kp_rot", 5.0,
            ParameterDescriptor(
                description="force_control 자세유지 강성(N·m/rad)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=50.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "force_ramp_time", 1.0,
            ParameterDescriptor(
                description="force_des 0→목표 무충격 램프 시간(s)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.0)],
            ),
        )
        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.linear_step = self.get_parameter("linear_step").value
        self.angular_step = self.get_parameter("angular_step").value
        self.urdf_path = self.get_parameter("urdf_path").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.gravity_scale = [self.get_parameter(f"gravity_scale_{j}").value for j in range(1, 8)]
        self.gravity_kd = [self.get_parameter(f"gravity_kd_{j}").value for j in range(1, 8)]
        self.probe_force_threshold = self.get_parameter("probe_force_threshold").value
        self.probe_speed = self.get_parameter("probe_speed").value
        self.probe_max_distance = self.get_parameter("probe_max_distance").value
        self.probe2_speed_max = self.get_parameter("probe2_speed_max").value
        self.probe2_speed_min = self.get_parameter("probe2_speed_min").value
        self.force_des = self.get_parameter("force_des").value
        self.force_max_distance = self.get_parameter("force_max_distance").value
        self.force_kp_lin = self.get_parameter("force_kp_lin").value
        self.force_kp_rot = self.get_parameter("force_kp_rot").value
        self.force_ramp_time = self.get_parameter("force_ramp_time").value

        self.enable_flag = False
        self.control_ready = False
        self.target = None          # [x,y,z,r,p,yaw] 목표 flange pose (없으면 스트리밍 보류)
        self.drag_mode_active = False
        self.T_ft = None            # flange→tool(end_point_link) 상수 변환 (자동보정, 1회)
        # probe_z 상태
        self.probe_active = False
        self.probe_phase = None      # 'approach' / 'retract'
        self.probe_mode = "const"    # 'const'(probe_z 일정스텝) / 'decel'(probe_z2 감속)
        self.probe_traveled = 0.0
        self.probe_start_target = None
        self.probe_result = None     # (success, message)
        self._probe_consec = 0
        # force_control 상태
        self.force_active = False
        self.force_entry_pose = None   # (R_des, p_des) 진입 첫 사이클에 현재 pose로 캡처
        self.force_entry_t = 0.0       # force_des 램프 기준 시각(perf_counter)

        self._init_arm()
        self._init_dynamics()

        self.joint_states_pub = self.create_publisher(JointState, "feedback/joint_states", 1)
        # 관성가중 토크-잔차 EE 힘 추정 (watt calculate_force와 동일 원리)
        self.ee_force_pub = self.create_publisher(WrenchStamped, "feedback/endeffector_force", 1)
        # move_mit이 함의하는 기대 토크 T_ref (move_p면 중력 기준선)
        self.expected_mit_torque_pub = self.create_publisher(
            JointState, "feedback/expected_mit_torque", 1
        )
        self.create_service(Trigger, "get_robot_state", self._get_state_cb)
        self.create_service(SetBool, "drag_mode", self._drag_cb)
        self.create_service(Trigger, "probe_z", self._probe_cb)
        self.create_service(Trigger, "probe_z2", self._probe2_cb)
        self.create_service(SetBool, "force_control", self._force_cb)
        self.add_on_set_parameters_callback(self._on_set_params)

        # 모든 CAN I/O(읽기+move_p)는 이 한 스레드에서만 → 레이스 방지
        self.loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.loop_thread.start()
        # 키보드 입력 스레드 (target만 갱신, CAN 접근 없음)
        self.kbd_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.kbd_thread.start()

        self.get_logger().info(KEYMAP_BANNER)

    ### initialization
    def _init_arm(self):
        # Nero, 펌웨어 v111 고정 (simple_node와 동일)
        config = create_agx_arm_config(
            robot=self.arm_type, comm="can", channel=self.can_port,
            firmeware_version=NeroFW.V111,
        )
        self.agx_arm = AgxArmFactory.create_arm(config)
        self.agx_arm.connect()
        self.arm_joint_names = list(config["joint_limits"].keys())
        self.arm_joint_count = self.agx_arm.joint_nums

        if self.auto_enable and not self._enable_arm(True):
            self.get_logger().error("Failed to auto-enable the arm")
        self.agx_arm.set_speed_percent(self.speed_percent)

    def _init_dynamics(self):
        """중력보상 모델(Pinocchio) 로드. 실패해도 노드는 정상 동작(drag만 비활성).
        simple_node에서 차용."""
        self._gc_ok = False
        self.pin_model = None
        self.ee_frame_id = None
        if not _HAS_PIN:
            self.get_logger().warn("pinocchio not available; drag(gravity-comp) disabled")
            return
        if not self.urdf_path:
            self.get_logger().warn("urdf_path empty; drag(gravity-comp) disabled")
            return
        try:
            self.pin_model = pin.buildModelFromUrdf(self.urdf_path)
            self.pin_data = self.pin_model.createData()
            if self.pin_model.nv != self.arm_joint_count:
                self.get_logger().warn(
                    f"URDF nv({self.pin_model.nv}) != joints({self.arm_joint_count}); check joint order"
                )
            self._gc_ok = True
            self.get_logger().info(f"Gravity model loaded (nv={self.pin_model.nv})")
            self._resolve_ee_frame(self.ee_frame)
        except Exception as e:
            self.get_logger().error(f"gravity model load failed: {e}")

    def _enable_arm(self, enable=True):
        start = time.time()
        while not (self.agx_arm.enable() if enable else self.agx_arm.disable()):
            if time.time() - start > self.enable_timeout:
                self.get_logger().error(f"Timeout to {'enable' if enable else 'disable'} arm")
                return False
            time.sleep(1)
        self.enable_flag = enable
        self.get_logger().info(f"Arm {'enabled' if enable else 'disabled'}")
        return True

    ### helpers
    def _to_ros_time(self, ts):
        t = Time()
        t.sec = int(ts)
        t.nanosec = int((ts - t.sec) * 1e9)
        return t

    def _arm_ready(self):
        js = self.agx_arm.get_joint_angles()
        return js is not None and js.hz > 0

    def _read_flange_pose(self):
        """현재 flange pose [x,y,z,r,p,yaw] 또는 None."""
        fp = self.agx_arm.get_flange_pose()
        if fp is None or fp.hz <= 0:
            return None
        return list(fp.msg)

    @staticmethod
    def _wrap_pi(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _sanitize(self, pose):
        """move_p 자세 범위로 정리: roll/yaw∈[-π,π], pitch∈(-π/2,π/2)."""
        x, y, z, r, p, yw = pose
        r = self._wrap_pi(r)
        yw = self._wrap_pi(yw)
        eps = 1e-3
        p = max(-HALF_PI + eps, min(HALF_PI - eps, p))
        return [x, y, z, r, p, yw]

    def _ensure_flange_tool(self, q):
        """flange↔tool(end_point_link) 상수 변환 1회 산출 (루프 스레드).
        SDK flange 포즈 + Pinocchio tool FK 자동보정 — SDK flange가 URDF 어느 링크인지 몰라도 됨."""
        if self.T_ft is not None or not self._gc_ok or self.ee_frame_id is None:
            return
        fp = self._read_flange_pose()
        if fp is None:
            return
        pin.framesForwardKinematics(self.pin_model, self.pin_data, np.asarray(q, dtype=float))
        oMf = self.pin_data.oMf[self.ee_frame_id]
        T_bt = (np.array(oMf.rotation), np.array(oMf.translation))      # base→tool (Pinocchio)
        T_bf = (_euler_to_R(fp[3:6]), np.asarray(fp[:3], dtype=float))  # base→flange (SDK)
        self.T_ft = _compose(_inverse(T_bf), T_bt)                     # flange→tool 상수
        Rft, pft = self.T_ft
        self.get_logger().info(
            f"flange->tool calib: |p|={np.linalg.norm(pft):.4f} m  rpy={np.round(_R_to_euler(Rft), 3)}"
        )

    def _jog_tool_frame(self, F, axis, sign):
        """flange pose F에 EE(tool) 프레임 jog 1스텝 적용 → 새 flange pose 리스트.
        이동(axis<3): tool 축 직선. 회전(axis>=3): tool 팁 둘레. 순수 math(스레드 안전)."""
        T_bf = (_euler_to_R(F[3:6]), np.asarray(F[:3], dtype=float))
        T_bt = _compose(T_bf, self.T_ft)                    # base→tool
        e = np.zeros(3)
        e[axis % 3] = float(sign)
        if axis < 3:                                        # 이동: tool 축
            dT = (np.eye(3), e * self.linear_step)
        else:                                               # 회전: tool 팁 둘레
            dT = (_rotvec_to_R(e * self.angular_step), np.zeros(3))
        T_bt_new = _compose(T_bt, dT)                       # 우곱 = 툴(바디) 델타
        Rm, p = _compose(T_bt_new, _inverse(self.T_ft))     # 다시 base→flange
        return [float(p[0]), float(p[1]), float(p[2]), *[float(a) for a in _R_to_euler(Rm)]]

    def _step_tool_z(self, F, dist):
        """flange pose F를 tool z축으로 dist[m] 평행이동(회전 불변) → 새 flange pose 리스트."""
        R_bt = _euler_to_R(F[3:6]) @ self.T_ft[0]           # tool 회전(base)
        zdir = R_bt @ np.array([0.0, 0.0, 1.0])
        p = np.asarray(F[:3], dtype=float) + float(dist) * zdir
        return [float(p[0]), float(p[1]), float(p[2]), F[3], F[4], F[5]]

    def _probe_step(self, F):
        """probe 상태기계(루프 스레드). tool -z 전진 → |F_z|≥임계 접촉 → 전진거리만큼 복귀.
        F: EE프레임 추정힘(없으면 None→그 주기 미접촉). 모든 move_p는 이 루프 스레드에서만.
        probe_mode='const'(probe_z)는 일정 스텝, 'decel'(probe_z2)은 목표 근처로 갈수록 감속."""
        fz = abs(float(F[2])) if F is not None else 0.0
        if self.probe_phase == "approach":
            self._probe_consec = (self._probe_consec + 1) if fz >= self.probe_force_threshold else 0
            if self._probe_consec >= PROBE_CONSEC:
                self.probe_result = (True, f"contact Fz={fz:.2f}N @ {self.probe_traveled:.4f}m")
                self.probe_phase = "retract"
            elif self.probe_traveled >= self.probe_max_distance:
                self.probe_result = (False, f"no contact within {self.probe_max_distance:.3f}m")
                self.probe_phase = "retract"
            else:
                step = self._probe_approach_step()                # 'decel'이면 감속, 'const'면 일정
                self.target = self._sanitize(self._step_tool_z(self.target, -step))   # -z 전진
                self.probe_traveled += step
        elif self.probe_phase == "retract":
            if self.probe_traveled <= 1e-9:
                self.target = list(self.probe_start_target)   # 정확 복귀(잔차 제거)
                self.probe_active = False
                self.probe_phase = None
            else:
                back = min(self._probe_retract_step(), self.probe_traveled)
                self.target = self._sanitize(self._step_tool_z(self.target, +back))
                self.probe_traveled -= back
        if self.target is not None:
            self.agx_arm.move_p(list(self.target))            # 루프 스레드 CAN

    def _probe_approach_step(self):
        """approach 1주기 전진 스텝(m). 'const'=일정(probe_speed),
        'decel'=남은거리 비례 선형 감속(시작 speed_max → 목표 speed_min). speed_min>0 하한이
        목표(probe_max_distance) 도달·정상 종료를 보장."""
        rate = max(1, self.pub_rate)
        if self.probe_mode == "decel":
            remaining = max(0.0, self.probe_max_distance - self.probe_traveled)
            frac = remaining / self.probe_max_distance if self.probe_max_distance > 1e-9 else 0.0
            v = self.probe2_speed_min + (self.probe2_speed_max - self.probe2_speed_min) * frac
            return v / rate
        return self.probe_speed / rate

    def _probe_retract_step(self):
        """retract 1주기 복귀 스텝(m). 'decel'은 빠른 복귀(speed_max), 'const'는 probe_speed."""
        rate = max(1, self.pub_rate)
        v = self.probe2_speed_max if self.probe_mode == "decel" else self.probe_speed
        return v / rate

    ### control loop (sole CAN owner)
    def _loop(self):
        while rclpy.ok():
            t0 = time.perf_counter()
            try:
                self._loop_once()
            except Exception as e:
                self.get_logger().warn(f"loop error: {e}", throttle_duration_sec=1.0)
            dt = 1.0 / max(1, self.pub_rate) - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def _loop_once(self):
        if not self.agx_arm.is_ok():
            return
        if not self.control_ready and self._arm_ready():
            self.control_ready = True
            self.get_logger().info("Agx_arm feedback is ready, control enabled")

        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            return
        q = list(js.msg)

        # 관절별 모터 상태(velocity/effort) — joint_states + force 추정 공용 (get_motor_states 1회)
        velocities, efforts = [], []
        for j in range(1, self.arm_joint_count + 1):
            ms = self.agx_arm.get_motor_states(j)
            velocities.append(ms.msg.velocity if ms is not None else 0.0)
            efforts.append(ms.msg.torque if ms is not None else 0.0)

        # feedback/joint_states (position + velocity + effort)
        msg = JointState()
        msg.header.stamp = self._to_ros_time(js.timestamp)
        msg.name = list(self.arm_joint_names)
        msg.position = q
        msg.velocity = velocities
        msg.effort = efforts
        self.joint_states_pub.publish(msg)

        # 중력 G(q): expected_mit_torque + endeffector_force 공용 (gc_ok 시 매 주기)
        F = None
        if self._gc_ok:
            self._ensure_flange_tool(q)        # flange↔tool 변환 1회 보정(EE축 jog용)
            tau_g = pin.computeGeneralizedGravity(
                self.pin_model, self.pin_data, np.asarray(q, dtype=float)
            )

            # expected_mit_torque: move_mit 기대 T_ref (move_p면 중력 기준선)
            em = JointState()
            em.header.stamp = msg.header.stamp
            em.name = list(self.arm_joint_names)
            em.effort = self._expected_mit_torque(tau_g, velocities)
            self.expected_mit_torque_pub.publish(em)

            # endeffector_force: 관성가중 토크-잔차로 EE 힘 추정 (drag/move_p 무관)
            if self.ee_frame_id is not None:
                F = self._estimate_ee_force(q, efforts, tau_g)
                if F is not None:
                    w = WrenchStamped()
                    w.header.stamp = msg.header.stamp
                    w.header.frame_id = self.ee_frame
                    w.wrench.force.x = float(F[0])
                    w.wrench.force.y = float(F[1])
                    w.wrench.force.z = float(F[2])
                    self.ee_force_pub.publish(w)

        # probe_z: tool -z 전진 → 접촉 → 복귀 상태기계 (teleop·drag와 배타)
        if self.probe_active:
            self._probe_step(F)
            return

        # drag(중력보상) 모드: move_p 스트리밍 대신 move_mit 중력보상만
        if self.drag_mode_active and self._gc_ok:
            self._drag_once(q)
            return

        # force_control(힘제어 유지): move_p 대신 move_mit 토크(중력 + 작업렌치)
        if self.force_active and self._gc_ok and self.ee_frame_id is not None:
            self._force_servo(q)
            return

        # target 최초/드래그 해제 후 시드: 현재 flange pose
        if self.target is None:
            seed = self._read_flange_pose()
            if seed is None:
                return                      # flange pose 아직 → 스트리밍 보류
            self.target = self._sanitize(seed)
            self.get_logger().info(
                "target seeded: [" + ", ".join(f"{v:+.3f}" for v in self.target) + "]"
            )

        # 매 루프 move_p 스트리밍 (원자적 스냅샷)
        target = self.target
        self.agx_arm.move_p(list(target))

    def _drag_once(self, q):
        """중력보상 drag: 관절별 move_mit(kp=0, kd=gravity_kd, t_ff=gravity_scale*G(q)).
        simple_node에서 차용. target을 None으로 비워 drag 해제 시 현재 flange pose로 재시드."""
        self.target = None
        tau = pin.computeGeneralizedGravity(
            self.pin_model, self.pin_data, np.asarray(q, dtype=float)
        )
        for i in range(self.arm_joint_count):
            self.agx_arm.move_mit(
                joint_index=i + 1, p_des=0.0, v_des=0.0,
                kp=0.0, kd=self.gravity_kd[i],
                t_ff=float(self.gravity_scale[i] * tau[i]),
            )

    def _force_servo(self, q):
        """tool -z로 force_des[N]를 feedforward로 눌러 유지 + 나머지 축 위치/자세 유지.
        move_mit 토크(kp=0, 모터측 kd 감쇠)로 송신. 루프 스레드 전용.

        설계 메모 (V111 / 50Hz 제약):
        - 힘 피드백 없음: 토크모드에선 토크-잔차 힘추정이 자기 명령을 되읽으므로(평형에서
          측정토크=명령토크) feedforward로 누른다. 표면 반작용이 force_des와 평형 → 접촉력=force_des.
          무충격 위해 force_des는 smoothstep 램프.
        - 호스트 속도 피드백 없음(V111 get_motor_states velocity=0): 모든 감쇠는 move_mit kd(모터측).
        - 위치/자세 유지는 호스트 강성(P항)만. 힘 축 성분은 제거해 위치-힘 충돌 방지.
        - push 방향·자세 기준은 ee_frame(=force 추정 프레임)으로 통일.
        """
        qn = np.asarray(q, dtype=float)
        G = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, qn)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, qn)
        oMf = self.pin_data.oMf[self.ee_frame_id]
        R, p = np.array(oMf.rotation), np.array(oMf.translation)

        # 진입 첫 사이클: 현재 pose를 기준으로 캡처(오차 0 출발) + 램프 기준 시각
        if self.force_entry_pose is None:
            self.force_entry_pose = (R.copy(), p.copy())
            self.force_entry_t = time.perf_counter()
            self.get_logger().info("force_control: entry pose captured, ramping 0 -> force_des")
        R_des, p_des = self.force_entry_pose

        # force_des 무충격 램프 (smoothstep: 0속도 시단/종단)
        frac = max(0.0, min(1.0, (time.perf_counter() - self.force_entry_t)
                            / max(1e-3, self.force_ramp_time)))
        f_now = self.force_des * (3.0 * frac * frac - 2.0 * frac ** 3)

        # LOCAL_WORLD_ALIGNED: twist/wrench를 EE점·월드축으로 → 위치/자세 오차와 정합
        J = pin.computeFrameJacobian(self.pin_model, self.pin_data, qn,
                                     self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        push = -(R @ np.array([0.0, 0.0, 1.0]))            # tool -z (base) = 누름/전진 방향

        F_lin = self.force_kp_lin * (p_des - p)            # 위치 유지(호스트 속도항 없음)
        F_lin = F_lin - float(np.dot(F_lin, push)) * push  # 힘 축 성분 제거(위치-힘 충돌 방지)
        F_lin = F_lin + push * f_now                       # feedforward 누름힘(램프)
        M_rot = self.force_kp_rot * _Rot.from_matrix(R_des @ R.T).as_rotvec()  # 자세 유지(base)

        wrench = np.concatenate([F_lin, M_rot])            # 6D, base/LWA
        g_scaled = np.array([self.gravity_scale[i] * G[i] for i in range(self.arm_joint_count)])
        tau = np.clip(g_scaled + J.T @ wrench, -16.0, 16.0)  # ±16 N·m = t_ff 12-bit 인코딩 한계

        for i in range(self.arm_joint_count):
            self.agx_arm.move_mit(
                joint_index=i + 1, p_des=0.0, v_des=0.0,
                kp=0.0, kd=self.gravity_kd[i], t_ff=float(tau[i]),   # kd = 모터측 감쇠
            )

        # 안전: 기준에서 누름축으로 force_max_distance 초과 전진(무접촉 폭주) 시 정지
        if float(np.dot(p - p_des, push)) > self.force_max_distance:
            self.get_logger().warn(
                f"force_control: travel > {self.force_max_distance:.3f}m (no contact?) -> stop"
            )
            self.force_active = False
            self.force_entry_pose = None
            self.target = None                              # move_p 복귀(현재 pose 재시드)

    ### endeffector force estimation
    def _resolve_ee_frame(self, name):
        """ee_frame 이름 → Pinocchio frame id 해석/검증. init·런타임 변경 공용."""
        if not self._gc_ok:
            self.ee_frame_id = None
            return
        if self.pin_model.existFrame(name):
            self.ee_frame = name
            self.ee_frame_id = self.pin_model.getFrameId(name)
            self.get_logger().info(f"endeffector_force frame = '{name}'")
        else:
            self.ee_frame_id = None
            self.get_logger().warn(f"ee_frame '{name}' not in model; endeffector_force disabled")

    def _estimate_ee_force(self, q, efforts, tau):
        """관성가중 토크-잔차로 EE 힘(3D, EE 좌표) 추정. watt calculate_force와 동일 원리.
        준정적 가정. np.ndarray(3,) 반환; 불가/특이점이면 None.
        try/except·isfinite는 값 가드가 아니라, 특이점 발산이 _loop_once의
        모터 명령까지 죽이지 않도록 하는 예외 격리(비특이 자세 값은 watt와 동일)."""
        if not (self._gc_ok and tau is not None and self.ee_frame_id is not None):
            return None
        try:
            qnp = np.asarray(q, dtype=float)
            J = pin.computeFrameJacobian(self.pin_model, self.pin_data, qnp,
                                         self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            Jv = J[:3, :]                                   # 3×7 선형(EE 좌표)
            pin.crba(self.pin_model, self.pin_data, qnp)    # data.M (상삼각만 채움)
            M = np.triu(self.pin_data.M)
            M = M + M.T - np.diag(np.diag(M))               # 대칭화 (crba gotcha)
            Minv = np.linalg.inv(M)
            J_bar = Minv @ Jv.T @ np.linalg.inv(Jv @ Minv @ Jv.T)   # 7×3, 관성가중
            F = J_bar.T @ (np.asarray(efforts) - np.asarray(tau))   # 측정 − 중력 → 3D 힘
        except np.linalg.LinAlgError:
            return None                                     # 정확 특이 → 그 사이클 발행만 스킵
        if not np.all(np.isfinite(F)):
            return None                                     # 근처 특이 inf/nan → 스킵
        return F

    def _expected_mit_torque(self, tau, velocities):
        """move_mit이 함의하는 기대 토크 T_ref = kp(p_des−q)+kd(v_des−v)+t_ff 리스트.
        drag(kp=0): kd*(0−v) + gravity_scale*G(q) (= _drag_once가 보내는 명령).
        move_p(MIT 미사용): G(q) 기준선."""
        if self.drag_mode_active:
            return [float(self.gravity_kd[i] * (0.0 - velocities[i])
                          + self.gravity_scale[i] * tau[i])
                    for i in range(self.arm_joint_count)]
        return [float(v) for v in tau]                      # move_p: 중력 기준선

    ### keyboard
    def _keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().error(
                "stdin is not a TTY; keyboard teleop disabled. Run with `ros2 run` in a terminal."
            )
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x03" or ch == "\x1b":   # Ctrl-C / ESC
                    self.get_logger().info("quit key received")
                    rclpy.shutdown()
                    break
                self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, ch):
        if self.target is None or self.probe_active or self.force_active:
            return                              # 미시드 / probe·force 동작 중 → teleop 차단
        if ch in JOG_KEYS:
            axis, sign = JOG_KEYS[ch]
            if self.T_ft is None:
                # 폴백: flange↔tool 미보정(Pinocchio 없음 등) → 기존 base 증분
                step = self.linear_step if axis < 3 else self.angular_step
                t = list(self.target)
                t[axis] += sign * step
                self.target = self._sanitize(t)
            else:
                # EE(tool) 프레임 jog → 새 flange pose (통째 교체: GIL 하 원자적)
                self.target = self._sanitize(self._jog_tool_frame(self.target, axis, sign))
        elif ch == " ":
            seed = self._read_flange_pose()
            if seed is not None:
                self.target = self._sanitize(seed)
                self.get_logger().info("target re-synced to current flange pose")
        elif ch == "[":
            self.linear_step = max(1e-4, self.linear_step / 1.5)
            self.get_logger().info(f"linear_step={self.linear_step:.4f} m")
        elif ch == "]":
            self.linear_step = min(0.1, self.linear_step * 1.5)
            self.get_logger().info(f"linear_step={self.linear_step:.4f} m")
        elif ch == "-":
            self.angular_step = max(1e-4, self.angular_step / 1.5)
            self.get_logger().info(f"angular_step={self.angular_step:.4f} rad")
        elif ch == "=":
            self.angular_step = min(0.5, self.angular_step * 1.5)
            self.get_logger().info(f"angular_step={self.angular_step:.4f} rad")
        elif ch == "p":
            actual = self._read_flange_pose()
            tgt = "[" + ", ".join(f"{v:+.3f}" for v in self.target) + "]"
            act = ("[" + ", ".join(f"{v:+.3f}" for v in actual) + "]") if actual else "unavailable"
            self.get_logger().info(f"target={tgt}  actual={act}")

    def _drag_cb(self, request, response):
        """중력보상 drag on/off (SetBool). simple_node에서 차용.
        on이면 _loop_once가 move_p 대신 중력보상 move_mit를 송신, off면 move_p 복귀."""
        if request.data and not self._gc_ok:
            response.success = False
            response.message = "gravity model not loaded; cannot enter drag"
            self.get_logger().warn(response.message)
            return response
        if request.data and not self.enable_flag:
            response.success = False
            response.message = "enable arm before drag"
            self.get_logger().warn(response.message)
            return response
        if request.data and self.probe_active:
            response.success = False
            response.message = "probe in progress; cannot enter drag"
            self.get_logger().warn(response.message)
            return response
        if request.data and self.force_active:
            response.success = False
            response.message = "force_control active; cannot enter drag"
            self.get_logger().warn(response.message)
            return response
        self.drag_mode_active = bool(request.data)
        if not self.drag_mode_active:
            self.target = None          # 해제: 현재 flange pose로 재시드 후 move_p 복귀
        response.success = True
        response.message = f"Drag(gravity-comp) {'on' if request.data else 'off'}"
        self.get_logger().info(response.message)
        return response

    def _force_cb(self, request, response):
        """force_control 유지 모드 on/off (SetBool, 비블로킹).
        on이면 _loop_once가 move_p 대신 move_mit 토크(중력 + 작업렌치)로 tool -z를 force_des로 누름.
        진입 사전조건: enable + 중력모델 + ee_frame, drag/probe와 배타.
        off면 현재 flange pose로 재시드해 move_p 복귀."""
        if request.data:
            if not self.enable_flag:
                response.success, response.message = False, "enable arm before force_control"
                self.get_logger().warn(response.message)
                return response
            if not (self._gc_ok and self.ee_frame_id is not None):
                response.success, response.message = False, "force model/frame not ready"
                self.get_logger().warn(response.message)
                return response
            if self.drag_mode_active or self.probe_active:
                response.success, response.message = (
                    False, "drag/probe active; cannot enter force_control")
                self.get_logger().warn(response.message)
                return response
            self.force_entry_pose = None        # 루프 첫 사이클에서 현재 pose로 캡처
            self.force_active = True            # ★ 마지막에 set (루프 스레드 인계)
        else:
            self.force_active = False
            self.force_entry_pose = None
            self.target = None                  # 해제: 현재 flange pose로 재시드 → move_p 복귀
        response.success = True
        response.message = (f"Force-hold on (force_des={self.force_des:.1f}N)"
                            if request.data else "Force-hold off")
        self.get_logger().info(response.message)
        return response

    def _probe_cb(self, request, response):
        """probe_z (Trigger): tool -z 일정 스텝 전진 → |F_z|≥임계 접촉 → 전진거리만큼 복귀."""
        return self._run_probe("const", "probe_z", response)

    def _probe2_cb(self, request, response):
        """probe_z2 (Trigger): probe_z와 동일하되 목표(probe_max_distance) 근처로 갈수록
        스텝을 감속(speed_max→speed_min)시켜 처음엔 빠르게·끝엔 부드럽게 접근."""
        return self._run_probe("decel", "probe_z2", response)

    def _run_probe(self, mode, name, response):
        """probe_z / probe_z2 공통: 사전조건 검사 → 셋업 → 완료까지 블로킹.
        동작 중 teleop·drag 차단. force 추정(Pinocchio + ee_frame + T_ft) 필요.
        mode='const'(일정 스텝) / 'decel'(감속). 모든 move_p는 루프 스레드가 송신."""
        if not self.enable_flag:
            response.success, response.message = False, "enable arm before probe"
            return response
        if not (self._gc_ok and self.ee_frame_id is not None) or self.T_ft is None:
            response.success, response.message = False, "force model/frame not ready"
            return response
        if self.drag_mode_active:
            response.success, response.message = False, "disable drag before probe"
            return response
        if self.force_active:
            response.success, response.message = False, "disable force_control before probe"
            return response
        if self.probe_active:
            response.success, response.message = False, "probe already running"
            return response
        if self.target is None:
            response.success, response.message = False, "target not seeded yet"
            return response
        # 셋업 후 루프 스레드가 인계 (probe_active를 마지막에 set)
        self.probe_mode = mode
        self.probe_start_target = list(self.target)
        self.probe_traveled = 0.0
        self._probe_consec = 0
        self.probe_result = None
        self.probe_phase = "approach"
        self.probe_active = True
        slow = self.probe2_speed_min if mode == "decel" else self.probe_speed
        timeout = 2.0 * self.probe_max_distance / max(slow, 1e-3) + 10.0
        t0 = time.time()
        while self.probe_active and rclpy.ok():
            if time.time() - t0 > timeout:
                self.probe_active = False
                response.success, response.message = False, "probe timeout"
                self.get_logger().warn(f"{name}: timeout")
                return response
            time.sleep(0.02)
        ok, msg = self.probe_result or (False, "probe ended")
        response.success, response.message = ok, msg
        self.get_logger().info(f"{name}: {msg}")
        return response

    def _get_state_cb(self, request, response):
        """현재 로봇/모터 상태 조회 (Trigger). simple_node의 get_robot_state에서 차용.
        관절별 motor_states(position/velocity/current/torque)를 보고한다. CAN 읽기 전용
        (캐시 데이터 조회)이라 루프 스레드의 move_p 송신과 충돌하지 않는다."""
        lines = [
            f"enabled={self.enable_flag}  control_ready={self.control_ready}  "
            f"pub_rate={self.pub_rate}Hz  speed={self.speed_percent}%",
            f"step: linear={self.linear_step:.4f}m  angular={self.angular_step:.4f}rad",
            f"drag={self.drag_mode_active}  gc_ok={self._gc_ok}"
            + "  scale=[" + ",".join(f"{s:.2f}" for s in self.gravity_scale) + "]"
            + "  kd=[" + ",".join(f"{k:.2f}" for k in self.gravity_kd) + "]",
            f"force={self.force_active}  force_des={self.force_des:.1f}N  "
            f"kp_lin={self.force_kp_lin:.0f}  kp_rot={self.force_kp_rot:.1f}  "
            f"max_dist={self.force_max_distance:.3f}m  ramp={self.force_ramp_time:.1f}s",
        ]
        if self.target is not None:
            lines.append("target    =[" + ", ".join(f"{v:+.3f}" for v in self.target) + "]")
        else:
            lines.append("target    =unseeded")
        actual = self._read_flange_pose()
        if actual is not None:
            lines.append("flange    =[" + ", ".join(f"{v:+.3f}" for v in actual) + "]")
        else:
            lines.append("flange    =unavailable")
        js = self.agx_arm.get_joint_angles()
        q = list(js.msg) if (js is not None and js.hz > 0) else None
        if q is not None:
            lines.append("q[rad]    =[" + ", ".join(f"{v:+.3f}" for v in q) + "]")
        else:
            lines.append("q[rad]    =unavailable")

        # 관절별 모터 상태: 위치/속도/전류/토크
        pos, vel, cur, tau = [], [], [], []
        for j in range(1, self.arm_joint_count + 1):
            ms = self.agx_arm.get_motor_states(j)
            pos.append(ms.msg.position if ms is not None else float("nan"))
            vel.append(ms.msg.velocity if ms is not None else float("nan"))
            cur.append(ms.msg.current if ms is not None else float("nan"))
            tau.append(ms.msg.torque if ms is not None else float("nan"))
        lines.append("motor pos =[" + ", ".join(f"{v:+.3f}" for v in pos) + "] rad")
        lines.append("motor vel =[" + ", ".join(f"{v:+.3f}" for v in vel) + "] rad/s")
        lines.append("motor cur =[" + ", ".join(f"{v:+.2f}" for v in cur) + "] A")
        lines.append("motor tau =[" + ", ".join(f"{v:+.2f}" for v in tau) + "] N.m")

        st = self.agx_arm.get_arm_status()
        if st is not None:
            lines.append(
                f"ctrl_mode={st.msg.ctrl_mode}  arm_status={st.msg.arm_status}  "
                f"motion_status={st.msg.motion_status}"
            )
        response.success = True
        response.message = "\n".join(lines)
        return response

    def _on_set_params(self, params):
        for p in params:
            if p.name == "pub_rate":
                self.pub_rate = max(1, int(p.value))
            elif p.name == "linear_step":
                self.linear_step = max(0.0, float(p.value))
            elif p.name == "angular_step":
                self.angular_step = max(0.0, float(p.value))
            elif p.name == "speed_percent":
                self.speed_percent = int(p.value)
                self.agx_arm.set_speed_percent(self.speed_percent)
            elif p.name == "ee_frame":
                self._resolve_ee_frame(p.value)
            elif p.name == "probe_force_threshold":
                self.probe_force_threshold = max(0.0, float(p.value))
            elif p.name == "probe_speed":
                self.probe_speed = max(0.0, float(p.value))
            elif p.name == "probe_max_distance":
                self.probe_max_distance = max(0.0, float(p.value))
            elif p.name == "probe2_speed_max":
                self.probe2_speed_max = max(0.0, float(p.value))
            elif p.name == "probe2_speed_min":
                self.probe2_speed_min = max(0.0, float(p.value))
            elif p.name == "force_des":
                self.force_des = max(0.0, float(p.value))
            elif p.name == "force_max_distance":
                self.force_max_distance = max(0.0, float(p.value))
            elif p.name == "force_kp_lin":
                self.force_kp_lin = max(0.0, float(p.value))
            elif p.name == "force_kp_rot":
                self.force_kp_rot = max(0.0, float(p.value))
            elif p.name == "force_ramp_time":
                self.force_ramp_time = max(0.0, float(p.value))
            elif p.name.startswith("gravity_scale_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.gravity_scale):
                    self.gravity_scale[idx] = max(0.0, min(1.5, float(p.value)))
            elif p.name.startswith("gravity_kd_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.gravity_kd):
                    self.gravity_kd[idx] = max(0.0, min(5.0, float(p.value)))
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(AgxArmTeleopWithMovePNode())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
