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
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 100)
        self.declare_parameter("enable_timeout", 5.0)
        self.declare_parameter("control_enabled", True)
        self.declare_parameter("debug", False)
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
        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.control_enabled = self.get_parameter("control_enabled").value
        self.debug = self.get_parameter("debug").value
        self.urdf_path = self.get_parameter("urdf_path").value
        self.gravity_scale = [self.get_parameter(f"gravity_scale_{j}").value for j in range(1, 8)]
        self.gravity_kd = [self.get_parameter(f"gravity_kd_{j}").value for j in range(1, 8)]

        self.enable_flag = False
        self.control_ready = False
        self.drag_mode_active = False
        self._was_drag = False
        self._last_sleep_ms = 0.0

        self._init_arm()
        self._init_dynamics()

        self.joint_states_pub = self.create_publisher(JointState, "feedback/joint_states", 1)
        self.create_subscription(JointState, "control/joint_states", self._control_cb, 1)
        self.create_service(SetBool, "enable_agx_arm", self._enable_cb)
        self.create_service(SetBool, "control_enable", self._gate_cb)
        self.create_service(SetBool, "drag_mode", self._drag_cb)
        self.create_service(Trigger, "get_robot_state", self._get_state_cb)
        self.add_on_set_parameters_callback(self._on_set_params)

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
        t_grav = t_mit = t_pub  # drag 아닐 때 grav/move_mit 구간은 0

        # 중력보상 drag: 모든 CAN 모션 명령은 이 스레드에서만 나간다(레이스 방지)
        if self.drag_mode_active and self._gc_ok:
            tau = pin.computeGeneralizedGravity(
                self.pin_model, self.pin_data, np.asarray(q, dtype=float)
            )
            tff = [float(self.gravity_scale[i] * tau[i]) for i in range(self.arm_joint_count)]
            t_grav = clk()
            for i in range(self.arm_joint_count):
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
        elif self._was_drag:
            # drag 해제 falling edge: 현재 자세를 move_j로 hold
            self.agx_arm.move_j(q)
            self._was_drag = False

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

        self.get_logger().info(f"{joints}", throttle_duration_sec=0.5)
        if self.fast_mode:
            self.agx_arm.move_js(joints)
        else:
            self.agx_arm.move_j(joints)

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
