#!/usr/bin/env python3
# -*-coding:utf8-*-
"""MoveIt 제어 전용 최소 노드.

control/joint_states 구독 → move_j/move_js, feedback/joint_states 발행.
"""
import time
import math
import threading
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool, Trigger
from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW

try:
    import pinocchio as pin
    _HAS_PIN = True
except ImportError:
    _HAS_PIN = False


class AgxArmSimpleNode(Node):

    def __init__(self):
        super().__init__("agx_arm_ctrl_simple_node")

        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "nero")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("fast_mode", False)
        # 평상시 제어 방식: "move_mit"(중력 피드포워드+PD 임피던스) | "move_j"(스무딩 위치제어)
        self.declare_parameter("arm_control_mode", "move_mit")
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 100)
        self.declare_parameter("enable_timeout", 5.0)
        self.declare_parameter("control_enabled", True)
        self.declare_parameter("debug", False)
        # 편차 가드: 명령(JTC 셋포인트) vs 실제 관절각 차이가 임계값을 window초 이상 넘으면
        # JTC goal 취소 + 중력보상 drag 진입
        self.declare_parameter("deviation_guard_enabled", False)
        self.declare_parameter("arm_controller_action", "arm_controller/follow_joint_trajectory")
        self.declare_parameter(
            "deviation_threshold", 0.80,
            ParameterDescriptor(
                description="명령-실제 관절각 최대 편차 임계값(rad)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.14, step=0.0)],
            ),
        )
        self.declare_parameter(
            "deviation_window_sec", 0.5,
            ParameterDescriptor(
                description="편차 임계 초과가 연속 지속돼야 트립하는 시간(s)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "deviation_cmd_timeout", 0.3,
            ParameterDescriptor(
                description="명령 신선도(s). 이보다 오래된 명령이면 가드 평가 안 함",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "deviation_torque_threshold", 2.0,
            ParameterDescriptor(
                description="측정 토크와 기대 피드포워드 토크(G(q)) 최대 차이 임계값(N.m)",
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=24.0, step=0.0)],
            ),
        )
        self.declare_parameter(
            "urdf_path",
            "/home/yunbeom/agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/nero_handeye.urdf",
        )
        # 조인트별 14개 파라미터: gravity_scale_1..7, gravity_kd_1..7
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
                    description=f"joint{j} 가상 점성감쇠",
                    floating_point_range=[FloatingPointRange(from_value=0.0, to_value=5.0, step=0.0)],
                ),
            )
        # move_mit 임피던스 게인(평상시 위치제어용, drag의 gravity_kd와 별개)
        for j in range(1, 8):
            self.declare_parameter(
                f"mit_kp_{j}", 10.0,
                ParameterDescriptor(
                    description=f"joint{j} 임피던스 위치강성 kp",
                    floating_point_range=[FloatingPointRange(from_value=0.0, to_value=500.0, step=0.0)],
                ),
            )
        for j in range(1, 8):
            self.declare_parameter(
                f"mit_kd_{j}", 0.8,
                ParameterDescriptor(
                    description=f"joint{j} 임피던스 감쇠 kd",
                    floating_point_range=[FloatingPointRange(from_value=0.0, to_value=5.0, step=0.0)],
                ),
            )
        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.arm_control_mode = self.get_parameter("arm_control_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.control_enabled = self.get_parameter("control_enabled").value
        self.debug = self.get_parameter("debug").value
        self.deviation_guard_enabled = self.get_parameter("deviation_guard_enabled").value
        self.arm_controller_action = self.get_parameter("arm_controller_action").value
        self.deviation_threshold = self.get_parameter("deviation_threshold").value
        self.deviation_window_sec = self.get_parameter("deviation_window_sec").value
        self.deviation_cmd_timeout = self.get_parameter("deviation_cmd_timeout").value
        self.deviation_torque_threshold = self.get_parameter("deviation_torque_threshold").value
        self.urdf_path = self.get_parameter("urdf_path").value
        self.gravity_scale = [self.get_parameter(f"gravity_scale_{j}").value for j in range(1, 8)]
        self.gravity_kd = [self.get_parameter(f"gravity_kd_{j}").value for j in range(1, 8)]
        self.mit_kp = [self.get_parameter(f"mit_kp_{j}").value for j in range(1, 8)]
        self.mit_kd = [self.get_parameter(f"mit_kd_{j}").value for j in range(1, 8)]

        self.enable_flag = False
        self.control_ready = False
        self.drag_mode_active = False
        self._was_drag = False
        self._last_sleep_ms = 0.0
        # 편차 가드 상태
        self.deviation_guard_on = self.deviation_guard_enabled  # 런타임 토글(파라미터와 별개)
        self._last_cmd = None        # 최근 명령 관절(arm_joint_names 순서)
        self._last_cmd_t = 0.0       # 최근 명령 수신 시각 time.time()
        self._dev_trip_since = None  # 편차 초과가 처음 시작된 perf_counter 시각

        self._init_arm()
        self._init_dynamics()
        if self.arm_control_mode == "move_mit" and not self._gc_ok:
            self.get_logger().warn(
                "gravity model not loaded; arm_control_mode falls back to move_j"
            )

        self.joint_states_pub = self.create_publisher(JointState, "feedback/joint_states", 1)
        # 현재 자세에서 필요한 중력 피드포워드 토크 G(q) (effort에 담음, drag 무관 상시 발행)
        self.ff_torque_pub = self.create_publisher(JointState, "feedback/feedforward_torque", 1)
        self.create_subscription(JointState, "control/joint_states", self._control_cb, 1)
        self.create_service(SetBool, "enable_agx_arm", self._enable_cb)
        self.create_service(SetBool, "control_enable", self._gate_cb)
        self.create_service(SetBool, "drag_mode", self._drag_cb)
        self.create_service(SetBool, "deviation_guard", self._deviation_guard_cb)
        self.create_service(Trigger, "get_robot_state", self._get_state_cb)
        self.add_on_set_parameters_callback(self._on_set_params)

        # 편차 가드: 트립 시 JTC(FollowJointTrajectory) goal 취소용 클라이언트
        self._fjt_cancel_cli = self.create_client(
            CancelGoal, f"{self.arm_controller_action}/_action/cancel_goal"
        )

        # 발행 전용 스레드: 제어 콜백(블로킹 move_*)과 독립적으로 feedback를 내보낸다.
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()

    ### initialization
    def _init_arm(self):
        # Nero, 펌웨어 v111 고정
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
        """중력보상 모델(Pinocchio) 로드. 실패해도 노드는 정상 동작(drag만 비활성)."""
        self._gc_ok = False
        self.pin_model = None
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

    def _can_control(self):
        if not self.control_ready:
            return False
        if self.drag_mode_active:
            return False
        if not self.enable_flag:
            self.get_logger().warn("Agx_arm is not enabled, cannot control")
            return False
        return self.control_enabled

    def _use_mit_control(self):
        """평상시 제어를 move_mit 임피던스로 할지. 중력모델 없으면 move_j로 폴백."""
        return self.arm_control_mode == "move_mit" and self._gc_ok

    ### publisher thread
    def _publish_loop(self):
        n, t0 = 0, time.time()
        while rclpy.ok():
            try:
                self._publish_once()
            except Exception as e:
                self.get_logger().warn(f"publish error: {e}")
            s0 = time.perf_counter()
            time.sleep(max(0.0, 1.0 / self.pub_rate))  # pub_rate 런타임 변경 반영
            self._last_sleep_ms = 1e3 * (time.perf_counter() - s0)
            if self.drag_mode_active:
                # drag 중 실제 달성 루프율(= move_mit 송신율) 1초마다 측정
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    if self.debug:
                        self.get_logger().info(f"drag loop rate: {n / (now - t0):.0f} Hz")
                    n, t0 = 0, now
            else:
                n, t0 = 0, time.time()

    def _publish_once(self):
        if not self.agx_arm.is_ok():
            return
        if not self.control_ready and self._arm_ready():
            self.control_ready = True
            self.get_logger().info("Agx_arm feedback is ready, control enabled")

        clk = time.perf_counter
        t0 = clk()
        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            return
        q = list(js.msg)
        t_js = clk()

        # velocity/effort는 관절별 motor_states에서 (single_node와 동일).
        # 읽기 실패 시 0.0으로 채워 발행/아래 drag 루프가 끊기지 않게 한다.
        velocities, efforts = [], []
        for j in range(1, self.arm_joint_count + 1):
            ms = self.agx_arm.get_motor_states(j)
            velocities.append(ms.msg.velocity if ms is not None else 0.0)
            efforts.append(ms.msg.torque if ms is not None else 0.0)
        t_ms = clk()

        msg = JointState()
        msg.header.stamp = self._to_ros_time(js.timestamp)
        msg.name = list(self.arm_joint_names)
        msg.position = q
        msg.velocity = velocities
        msg.effort = efforts
        self.joint_states_pub.publish(msg)
        t_pub = clk()

        # 중력토크 G(q): drag 여부와 무관하게 매 주기 계산·발행
        # (현재 자세에서 각 관절에 필요한 피드포워드 토크 → feedback/feedforward_torque.effort)
        tau = None
        if self._gc_ok:
            tau = pin.computeGeneralizedGravity(
                self.pin_model, self.pin_data, np.asarray(q, dtype=float)
            )
            ff = JointState()
            ff.header.stamp = msg.header.stamp
            ff.name = list(self.arm_joint_names)
            ff.effort = [float(v) for v in tau]
            self.ff_torque_pub.publish(ff)
        t_grav = clk()
        t_mit = t_grav  # drag 아닐 때 move_mit 구간은 0

        # 편차 가드: 측정 토크 vs 기대 피드포워드 토크 차이가 임계 초과로 지속되면 트립
        self._deviation_check(q, efforts, t_pub)

        # 모든 CAN 모션 명령은 이 스레드에서만 나간다(레이스 방지).
        # drag(kp=0)·정상 임피던스(kp>0)·move_j 모드 drag 해제 hold를 한 곳에서 처리.
        n = self.arm_joint_count
        if self.drag_mode_active and tau is not None:
            # drag: 중력보상만 (kp=0, 낮은 gravity_kd)
            tff = [float(self.gravity_scale[i] * tau[i]) for i in range(n)]
            for i in range(n):
                self.agx_arm.move_mit(
                    joint_index=i + 1, p_des=0.0, v_des=0.0,
                    kp=0.0, kd=self.gravity_kd[i],
                    t_ff=tff[i],
                )
            t_mit = clk()
            if self.debug:
                self.get_logger().info(
                    "t_ff=[" + ", ".join(f"{v:+.2f}" for v in tff) + "] N.m",
                    throttle_duration_sec=0.5,
                )
            self._was_drag = True
        else:
            if self._was_drag:
                # drag 막 해제: 현재 자세로 목표 고정(옛 명령으로 스냅 방지)
                self._last_cmd = list(q)
                if not self._use_mit_control():
                    self.agx_arm.move_j(q)          # move_j 모드 hold
                self._was_drag = False
            if self._use_mit_control() and tau is not None:
                # 정상 impedance 서보: kp>0, p_des=목표, t_ff=중력
                if self._last_cmd is None:
                    self._last_cmd = list(q)        # 첫 명령 전: 현재 자세 유지
                target = self._last_cmd             # 원자적 스냅샷
                m = min(n, len(target))
                for i in range(m):
                    self.agx_arm.move_mit(
                        joint_index=i + 1, p_des=float(target[i]), v_des=0.0,
                        kp=self.mit_kp[i], kd=self.mit_kd[i],
                        t_ff=float(self.gravity_scale[i] * tau[i]),
                    )
                t_mit = clk()
                if self.debug:
                    self.get_logger().info(
                        "mit p_des=[" + ", ".join(f"{v:+.2f}" for v in target[:m]) + "]",
                        throttle_duration_sec=0.5,
                    )

        if self.debug:
            work = 1e3 * (t_mit - t0)
            slp = self._last_sleep_ms  # 직전 사이클 sleep (pub_rate 고정이면 ≈ 현 사이클)
            self.get_logger().info(
                "[loop ms] "
                f"get_q={1e3 * (t_js - t0):.2f}  motor_states={1e3 * (t_ms - t_js):.2f}  "
                f"pub={1e3 * (t_pub - t_ms):.2f}  grav={1e3 * (t_grav - t_pub):.2f}  "
                f"move_mit={1e3 * (t_mit - t_grav):.2f}  work={work:.2f}  "
                f"sleep={slp:.2f}  cycle={work + slp:.2f}",
                throttle_duration_sec=0.5,
            )

    def _control_cb(self, msg: JointState):
        if not self._can_control():
            return
        cmd = dict(zip(msg.name, msg.position))
        joints = []
        for name in self.arm_joint_names:
            v = cmd.get(name, 0.0)
            joints.append(0.0 if (v is None or math.isnan(v)) else v)

        # 편차 가드용 최신 명령 기록(매 콜백 새 리스트 → GIL 하 참조 대입 원자적, Lock 불필요)
        self._last_cmd = joints
        self._last_cmd_t = time.time()

        self.get_logger().info(f"{joints}", throttle_duration_sec=0.5)
        if self._use_mit_control():
            return                      # 서보 루프(_publish_once)가 move_mit로 추종
        if self.fast_mode:
            self.agx_arm.move_js(joints)
        else:
            self.agx_arm.move_j(joints)

    ### deviation guard (편차 가드)
    def _deviation_check(self, q, efforts, t_now):
        """측정 토크 vs 기대 피드포워드 토크(중력모델 G(q)) 차이가 임계값을 window초 이상
        넘으면 트립. _publish_once에서 매 루프 호출. t_now는 perf_counter(monotonic).

        기대 토크는 drag 모드에서 적용하는 피드포워드와 동일: gravity_scale[i]*G(q)[i].
        외력(접촉/충돌)이 가해지면 위치 제어기가 맞서면서 측정 토크가 기대치에서 벌어진다.
        """
        if not self.deviation_guard_on or self.drag_mode_active:
            self._dev_trip_since = None
            return
        if not self.enable_flag or not self._gc_ok:        # enable + 중력모델 필요
            self._dev_trip_since = None
            return

        # ---- (구) 위치 기준 가드: 잠시 주석 처리 ----
        # if not self.control_enabled:
        #     self._dev_trip_since = None
        #     return
        # cmd, cmd_t = self._last_cmd, self._last_cmd_t      # 원자적 스냅샷
        # if cmd is None or (time.time() - cmd_t) > self.deviation_cmd_timeout:
        #     self._dev_trip_since = None                    # 명령 없음/오래됨(궤적 없음)
        #     return
        # if len(cmd) != len(q):                             # 길이 불일치 방어
        #     self._dev_trip_since = None
        #     return
        # max_err = max(abs(c - a) for c, a in zip(cmd, q))
        # tripped, trip_val = max_err > self.deviation_threshold, max_err

        # ---- 토크(피드포워드) 기준 가드 ----
        tau = pin.computeGeneralizedGravity(
            self.pin_model, self.pin_data, np.asarray(q, dtype=float)
        )
        expected = [float(self.gravity_scale[i] * tau[i]) for i in range(self.arm_joint_count)]
        n = min(len(efforts), len(expected))
        if n == 0:                                         # 토크 읽기 불가
            self._dev_trip_since = None
            return
        max_terr = max(abs(efforts[i] - expected[i]) for i in range(n))
        tripped, trip_val = max_terr > self.deviation_torque_threshold, max_terr

        if tripped:
            if self._dev_trip_since is None:
                self._dev_trip_since = t_now
            elif (t_now - self._dev_trip_since) >= self.deviation_window_sec:
                self._dev_trip_since = None
                self._deviation_trip(q, trip_val)
        else:
            self._dev_trip_since = None                    # 임계값 아래로 내려가면 디바운스 리셋

    def _deviation_trip(self, q, max_terr):
        self.get_logger().warn(
            f"Deviation guard tripped: max|tau_meas-tau_ff|={max_terr:.2f} N.m "
            f"> {self.deviation_torque_threshold:.2f} for {self.deviation_window_sec:.2f}s "
            "-> cancel trajectory + gravity-comp drag"
        )
        self._cancel_trajectory()                          # 안전 핵심: 명령 스트림 중단
        if self._gc_ok:
            self.drag_mode_active = True                   # 다음 루프부터 drag 분기가 중력보상 이어받음
            self.get_logger().info("Deviation guard: entered gravity-comp drag")
        else:                                              # 중력모델 없으면 drag 무의미 → hold 후 가드 끔
            self.get_logger().error(
                "gravity model not loaded; holding pose and disabling guard"
            )
            self.agx_arm.move_j(list(q))
            self.deviation_guard_on = False

    def _cancel_trajectory(self):
        """진행 중인 FollowJointTrajectory goal(들)을 취소한다.

        빈 goal_info = 모든 active goal 취소. MoveIt/RViz가 보낸 goal이라도 action server(JTC)에
        cancel 요청으로 취소되어, 트립 후 잔여 waypoint가 더 이상 흘러오지 않는다.
        """
        try:
            if not self._fjt_cancel_cli.service_is_ready():
                self.get_logger().warn("FJT cancel service not available, skip trajectory cancel")
                return
            self._fjt_cancel_cli.call_async(CancelGoal.Request())
        except Exception as e:
            self.get_logger().warn(f"FJT cancel failed: {e}")

    def _deviation_guard_cb(self, request, response):
        self.deviation_guard_on = bool(request.data)
        if not self.deviation_guard_on:
            self._dev_trip_since = None
        response.success = True
        response.message = f"Deviation guard {'on' if self.deviation_guard_on else 'off'}"
        self.get_logger().info(response.message)
        return response

    def _enable_cb(self, request, response):
        if not self._arm_ready():
            response.success = False
            response.message = "Agx_arm is not connected"
            return response
        response.success = self._enable_arm(request.data)
        if response.success:
            response.message = "Agx_arm enabled" if request.data else "Agx_arm disabled"
        else:
            response.message = "Failed to set enable state"
        return response

    def _gate_cb(self, request, response):
        self.control_enabled = request.data
        response.success = True
        response.message = f"Control gate {'opened' if request.data else 'closed'}"
        self.get_logger().info(response.message)
        return response

    def _drag_cb(self, request, response):
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
        self.drag_mode_active = bool(request.data)
        response.success = True
        response.message = f"Drag(gravity-comp) {'on' if request.data else 'off'}"
        self.get_logger().info(response.message)
        return response

    def _get_state_cb(self, request, response):
        """현재 로봇/노드 상태 조회 (Trigger)."""
        lines = [
            f"enabled={self.enable_flag}  control_ready={self.control_ready}  "
            f"control_gate={'open' if self.control_enabled else 'closed'}",
            f"drag={self.drag_mode_active}"
            + "  scale=[" + ",".join(f"{s:.2f}" for s in self.gravity_scale) + "]"
            + "  kd=[" + ",".join(f"{k:.2f}" for k in self.gravity_kd) + "]",
            f"dev_guard={'on' if self.deviation_guard_on else 'off'}(torque)  "
            f"tau_thr={self.deviation_torque_threshold:.2f}N.m  win={self.deviation_window_sec:.2f}s  "
            f"(pos_thr={self.deviation_threshold:.3f}rad disabled)",
            f"ctrl_method={self.arm_control_mode}"
            + (f" (mit active  kp=[{','.join(f'{v:.0f}' for v in self.mit_kp)}]"
               f"  kd=[{','.join(f'{v:.2f}' for v in self.mit_kd)}])"
               if self._use_mit_control() else " (mit inactive -> move_j)"),
        ]
        js = self.agx_arm.get_joint_angles()
        q = list(js.msg) if (js is not None and js.hz > 0) else None
        if q is not None:
            lines.append("q[rad]    =[" + ", ".join(f"{v:+.3f}" for v in q) + "]")
        else:
            lines.append("q[rad]    =unavailable")
        tau = []
        for j in range(1, self.arm_joint_count + 1):
            ms = self.agx_arm.get_motor_states(j)
            tau.append(ms.msg.torque if ms is not None else float("nan"))
        lines.append("tau_meas  =[" + ", ".join(f"{t:+.2f}" for t in tau) + "]")
        if self._gc_ok and q is not None:
            g = pin.computeGeneralizedGravity(
                self.pin_model, self.pin_data, np.asarray(q, dtype=float)
            )
            lines.append("G(q)model =[" + ", ".join(f"{t:+.2f}" for t in g) + "]")
            lines.append("# 부호: 관절별 tau_meas vs G(q)model 가 다르면 그 관절 t_ff 부호가 반대")
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
        """런타임 파라미터 변경. gravity_scale_j / gravity_kd_j / pub_rate 즉시 반영."""
        for p in params:
            if p.name.startswith("gravity_scale_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.gravity_scale):
                    self.gravity_scale[idx] = max(0.0, min(1.5, float(p.value)))
            elif p.name.startswith("gravity_kd_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.gravity_kd):
                    self.gravity_kd[idx] = max(0.0, min(5.0, float(p.value)))
            elif p.name == "pub_rate":
                self.pub_rate = max(10, int(p.value))
            elif p.name == "debug":
                self.debug = bool(p.value)
            elif p.name == "deviation_guard_enabled":
                self.deviation_guard_on = bool(p.value)
            elif p.name == "deviation_threshold":
                self.deviation_threshold = max(0.0, float(p.value))
            elif p.name == "deviation_window_sec":
                self.deviation_window_sec = max(0.0, float(p.value))
            elif p.name == "deviation_cmd_timeout":
                self.deviation_cmd_timeout = max(0.0, float(p.value))
            elif p.name == "deviation_torque_threshold":
                self.deviation_torque_threshold = max(0.0, float(p.value))
            elif p.name == "arm_control_mode":
                self.arm_control_mode = str(p.value)
            elif p.name.startswith("mit_kp_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.mit_kp):
                    self.mit_kp[idx] = max(0.0, min(500.0, float(p.value)))
            elif p.name.startswith("mit_kd_"):
                idx = int(p.name.rsplit("_", 1)[1]) - 1
                if 0 <= idx < len(self.mit_kd):
                    self.mit_kd[idx] = max(-5.0, min(5.0, float(p.value)))
        self.get_logger().info(
            "scale=[" + ",".join(f"{s:.2f}" for s in self.gravity_scale) + "] "
            "kd=[" + ",".join(f"{k:.2f}" for k in self.gravity_kd) + "] "
            f"pub_rate={self.pub_rate}"
        )
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(AgxArmSimpleNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
