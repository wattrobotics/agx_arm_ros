#!/usr/bin/env python3
# -*-coding:utf8-*-
"""Nero 실물 팔 ↔ ros2_control(topic 기반 하드웨어 인터페이스) 브리지.

ROS1 `innfos_node` 역할 노드.
  - command_topic(JTC 명령, topic HW의 write 발행) 구독 → Nero 제어
  - Nero 실측 → feedback_topic 발행 (topic HW의 read 구독 대상)
대응: innfos/input ↔ command_topic, innfos/output ↔ feedback_topic.

평상시 제어: arm_control_mode="move_mit"(기본)이면 JTC 셋포인트를 목표로 MIT 임피던스 서보
  (kp·kd + 중력 피드포워드 t_ff=gravity_scale·G(q))로 추종. "move_js"면 단순 position 스트리밍 폴백
  (중력모델 없으면 자동 폴백).

drag(중력보상) 모드(수동, 서비스 트리거):
  ON  : switch_controller로 arm_controller 비활성화 + 발행 스레드가 중력보상 MIT(kp=0, t_ff=G(q))로
        Nero 직접 구동. 그 동안 command 전달 차단.
  OFF : arm_controller 재활성화(JTC가 실측에서 이어받음) → 재활성화 완료 후 전달 재개. 해제 시점에
        현재 자세를 목표로 잡아(stale 명령 스냅 방지) 서보가 이어받음.

모든 CAN 모션 명령은 발행 스레드 한 곳에서만 나간다(레이스 방지). _command_cb는 목표만 기록.

토픽 메시지 타입은 sensor_msgs/JointState (joint_state_topic_hardware_interface/JointStateTopicSystem).
토픽 이름은 ros2_control xacro의 joint_commands_topic / joint_states_topic 과 일치해야 한다.
"""
import time
import math
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.msg import SetParametersResult
from agx_arm_msgs.msg import AgxArmStatus, JointDriveState, JointDriveStateArray
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW

try:
    import pinocchio as pin
    _HAS_PIN = True
except ImportError:
    _HAS_PIN = False


class NeroHardwareInterface(Node):

    def __init__(self):
        super().__init__("nero_hardware_interface")

        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "nero")
        self.declare_parameter("auto_enable", True)
        # 평상시 제어: "move_mit"(중력 피드포워드+PD 임피던스) | "move_js"(단순 위치 스트리밍 폴백)
        self.declare_parameter("arm_control_mode", "move_mit")
        # move_js 폴백 시 move_js(고속)/move_j 선택
        self.declare_parameter("fast_mode", True)
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 100)
        self.declare_parameter("enable_timeout", 5.0)
        # true면 제어 루프 각 구간 소요시간 로그(rqt/ros2 param set로 런타임 토글 가능)
        self.declare_parameter("debug", False)
        # ros2_control xacro의 joint_commands_topic / joint_states_topic 과 일치해야 함
        self.declare_parameter("command_topic", "control/joint_states")
        self.declare_parameter("feedback_topic", "feedback/joint_states")
        # drag(중력보상) 모드
        self.declare_parameter(
            "urdf_path",
            "/home/yunbeom/agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/nero_handeye.urdf",
        )
        self.declare_parameter("controller_manager_name", "controller_manager")
        self.declare_parameter("arm_controller_name", "arm_controller")
        for j in range(1, 8):
            self.declare_parameter(f"gravity_scale_{j}", 1.0)   # 0=무보상, 1=완전
        for j in range(1, 8):
            self.declare_parameter(f"gravity_kd_{j}", 0.05)     # drag 가상 점성감쇠
        # 평상시 MIT 임피던스 게인(drag의 gravity_kd와 별개)
        for j in range(1, 8):
            self.declare_parameter(f"mit_kp_{j}", 10.0)         # 위치강성
        for j in range(1, 8):
            self.declare_parameter(f"mit_kd_{j}", 0.8)          # 감쇠

        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.arm_control_mode = self.get_parameter("arm_control_mode").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.debug = self.get_parameter("debug").value
        self.command_topic = self.get_parameter("command_topic").value
        self.feedback_topic = self.get_parameter("feedback_topic").value
        self.urdf_path = self.get_parameter("urdf_path").value
        self.controller_manager_name = self.get_parameter("controller_manager_name").value
        self.arm_controller_name = self.get_parameter("arm_controller_name").value
        self.gravity_scale = [self.get_parameter(f"gravity_scale_{j}").value for j in range(1, 8)]
        self.gravity_kd = [self.get_parameter(f"gravity_kd_{j}").value for j in range(1, 8)]
        self.mit_kp = [self.get_parameter(f"mit_kp_{j}").value for j in range(1, 8)]
        self.mit_kd = [self.get_parameter(f"mit_kd_{j}").value for j in range(1, 8)]

        self.enable_flag = False
        # 실측을 1회 이상 수신해 발행을 시작했는지(= ROS1 innfos position_set 가드 역할).
        self.control_ready = False
        # drag 상태: drag_mode_active=제어 타이머가 중력보상 MIT 송신, _block_forward=command 전달 차단.
        # 둘을 분리해 OFF 전환(재활성화) 동안 stale 명령 누출을 막는다.
        self.drag_mode_active = False
        self._block_forward = False
        self._was_drag = False
        # MIT 서보 목표(arm_joint_names 순서). _command_cb가 기록, 제어 타이머가 추종.
        self._last_cmd = None

        self._init_arm()
        self._init_dynamics()
        if self.arm_control_mode == "move_mit" and not self._gc_ok:
            self.get_logger().warn("gravity model not loaded; arm_control_mode falls back to move_js")

        self.joint_states_pub = self.create_publisher(JointState, self.feedback_topic, 1)
        self.arm_status_pub = self.create_publisher(AgxArmStatus, "feedback/arm_status", 1)
        self.joint_drive_states_pub = self.create_publisher(
            JointDriveStateArray, "feedback/joint_drive_states", 1
        )
        self.create_subscription(JointState, self.command_topic, self._command_cb, 1)
        self.create_service(SetBool, "drag_mode", self._drag_cb)
        self._switch_cli = self.create_client(
            SwitchController, f"{self.controller_manager_name}/switch_controller"
        )

        # 제어 루프 = ROS 타이머. 블로킹 CAN(read/move_*)이 콜백(command/drag)을 막지 않도록
        # 전용 MutuallyExclusiveCallbackGroup에 두고 MultiThreadedExecutor로 돌린다(main 참고).
        # command_cb/drag는 default 그룹 → 제어 타이머와 다른 스레드에서 동시 실행.
        self._control_cbg = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            1.0 / self.pub_rate, self._control_tick, callback_group=self._control_cbg
        )
        # 런타임 파라미터 변경(rqt / ros2 param set) 반영
        self.add_on_set_parameters_callback(self._on_set_params)

    ### initialization
    def _init_arm(self):
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
        """중력보상 모델(Pinocchio) 로드. 실패해도 노드는 정상 동작(MIT→move_js 폴백)."""
        self._gc_ok = False
        self.pin_model = None
        if not _HAS_PIN:
            self.get_logger().warn("pinocchio not available; gravity comp disabled")
            return
        if not self.urdf_path:
            self.get_logger().warn("urdf_path empty; gravity comp disabled")
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

    def _use_mit_control(self):
        """평상시 제어를 MIT 임피던스로 할지. 중력모델 없으면 move_js로 폴백."""
        return self.arm_control_mode == "move_mit" and self._gc_ok

    def _on_set_params(self, params):
        """런타임 파라미터 변경(rqt / ros2 param set). 현재 debug만 즉시 반영."""
        for p in params:
            if p.name == "debug":
                self.debug = bool(p.value)
                self.get_logger().info(f"debug = {self.debug}")
        return SetParametersResult(successful=True)

    ### 제어 타이머: read → 계산 → 제어 송신(drag/정상 분기) → feedback 발행.
    ### 모든 CAN 모션은 이 타이머 콜백에서만 나간다(레이스 방지). _command_cb는 목표 저장만.
    def _control_tick(self):
        # 예외가 타이머/executor를 죽이지 않도록 격리
        try:
            self._control_once()
        except Exception as e:
            self.get_logger().warn(f"control loop error: {e}")

    def _control_once(self):
        clk = time.perf_counter
        t0 = clk()
        if not self.agx_arm.is_ok():
            return
        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            return
        if not self.control_ready:
            self.control_ready = True
            self.get_logger().info("Nero feedback is ready")
        t_js = clk()

        # 1. 상태 읽기 (motor high-spd + driver low-spd, 관절별)
        q = list(js.msg)
        velocities, efforts = [], []
        drive_msg = JointDriveStateArray()
        drive_msg.header.stamp = self._to_ros_time(js.timestamp)
        for idx in range(self.arm_joint_count):
            j = idx + 1
            ms = self.agx_arm.get_motor_states(j)
            ds = self.agx_arm.get_driver_states(j)
            velocities.append(ms.msg.velocity if ms is not None else 0.0)
            efforts.append(ms.msg.torque if ms is not None else 0.0)
            name = self.arm_joint_names[idx] if idx < len(self.arm_joint_names) else f"joint{j}"
            drive_msg.joints.append(self._build_joint_drive_state(name, ms, ds))
        t_read = clk()

        # 2. feedback 발행 (command와 무관하게 항상)
        msg = JointState()
        msg.header.stamp = self._to_ros_time(js.timestamp)
        msg.name = list(self.arm_joint_names)
        msg.position = q
        msg.velocity = velocities
        msg.effort = efforts
        self.joint_states_pub.publish(msg)
        self.joint_drive_states_pub.publish(drive_msg)
        self._publish_arm_status()
        t_pub = clk()

        # 3. 중력토크 G(q): drag/서보 양쪽 t_ff로 사용.
        tau = None
        if self._gc_ok:
            tau = pin.computeGeneralizedGravity(
                self.pin_model, self.pin_data, np.asarray(q, dtype=float)
            )
        t_grav = clk()

        # 4. 제어 송신 (drag / 정상 분기)
        n = self.arm_joint_count
        if self.drag_mode_active:
            # drag: 중력보상만 (kp=0)
            if tau is not None:
                for i in range(n):
                    self.agx_arm.move_mit(
                        joint_index=i + 1, p_des=0.0, v_des=0.0,
                        kp=0.0, kd=self.gravity_kd[i],
                        t_ff=float(self.gravity_scale[i] * tau[i]),
                    )
            self._was_drag = True
        else:
            if self._was_drag:
                # drag 막 해제: 현재 자세로 목표 고정(옛 명령으로 스냅 방지)
                self._last_cmd = list(q)
                self._was_drag = False
            if self._last_cmd is None:
                self._last_cmd = list(q)              # 첫 명령 전: 현재 자세 유지
            target = self._last_cmd                    # 원자적 스냅샷
            m = min(n, len(target))
            if self._use_mit_control() and tau is not None:
                # 정상 임피던스 서보: kp>0, p_des=목표, t_ff=중력
                for i in range(m):
                    self.agx_arm.move_mit(
                        joint_index=i + 1, p_des=float(target[i]), v_des=0.0,
                        kp=self.mit_kp[i], kd=self.mit_kd[i],
                        t_ff=float(self.gravity_scale[i] * tau[i]),
                    )
            else:
                # move_js 폴백(중력모델 없음 또는 arm_control_mode=move_js)
                joints = list(target[:m])
                if self.fast_mode:
                    self.agx_arm.move_js(joints)
                else:
                    self.agx_arm.move_j(joints)
        t_ctrl = clk()

        if self.debug:
            self.get_logger().info(
                "[loop ms] "
                f"get_q={1e3 * (t_js - t0):.2f}  read={1e3 * (t_read - t_js):.2f}  "
                f"pub={1e3 * (t_pub - t_read):.2f}  grav={1e3 * (t_grav - t_pub):.2f}  "
                f"ctrl={1e3 * (t_ctrl - t_grav):.2f}  work={1e3 * (t_ctrl - t0):.2f}",
                throttle_duration_sec=0.5,
            )

    def _build_joint_drive_state(self, name, ms, ds):
        """get_motor_states(ms) + get_driver_states(ds)를 JointDriveState 하나로 합친다.
        ms/ds가 None이면 해당 필드는 기본값(0/False)으로 둔다."""
        s = JointDriveState()
        s.joint_name = name
        if ms is not None:
            s.velocity = float(ms.msg.velocity)
            s.current = float(ms.msg.current)
            s.position = float(ms.msg.position)
            s.torque = float(ms.msg.torque)
        if ds is not None:
            s.voltage = float(ds.msg.vol)
            s.driver_temp = float(ds.msg.foc_temp)
            s.motor_temp = float(ds.msg.motor_temp)
            s.bus_current = float(ds.msg.bus_current)
            s.foc_status_code = int(ds.msg.foc_status_code)
            f = ds.msg.foc_status
            s.voltage_too_low = bool(f.voltage_too_low)
            s.motor_overheating = bool(f.motor_overheating)
            s.driver_overcurrent = bool(f.driver_overcurrent)
            s.driver_overheating = bool(f.driver_overheating)
            s.collision_status = bool(f.collision_status)
            s.driver_error_status = bool(f.driver_error_status)
            s.driver_enable_status = bool(f.driver_enable_status)
            s.stall_status = bool(f.stall_status)
        return s

    def _publish_arm_status(self):
        """Nero 상태(feedback/arm_status) 발행. single_node._publish_arm_status 포팅."""
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
            msg.joint_angle_limit.append(getattr(err, f"joint_{i+1}_angle_limit"))
            msg.communication_status_joint.append(getattr(err, f"communication_status_joint_{i+1}"))
        self.arm_status_pub.publish(msg)

    ### command 수신 (command_topic) → 목표 저장만 (제어/송신은 _control_once가 담당)
    def _command_cb(self, msg: JointState):
        # 실측 확정 + enable 전 / drag(전환 포함) 중에는 명령을 받지 않는다.
        if (not self.control_ready) or (not self.enable_flag) or self._block_forward:
            return

        if msg.name:
            cmd = dict(zip(msg.name, msg.position))
            joints = []
            for name in self.arm_joint_names:
                v = cmd.get(name)
                if v is None or math.isnan(v):
                    return                      # 불완전/NaN 명령 무시
                joints.append(float(v))
        else:
            if len(msg.position) < self.arm_joint_count:
                return
            joints = [float(msg.position[i]) for i in range(self.arm_joint_count)]
            if any(math.isnan(v) for v in joints):
                return

        # 목표만 기록(매 콜백 새 리스트 → GIL 하 참조 대입 원자적). 송신은 _control_once가.
        self._last_cmd = joints

    ### drag (수동, 서비스 트리거)
    def _switch_controller(self, activate, deactivate):
        """controller_manager에 activate/deactivate 요청. future 반환(서비스 없으면 None)."""
        if not self._switch_cli.service_is_ready():
            self.get_logger().warn(
                f"{self.controller_manager_name}/switch_controller not available"
            )
            return None
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.BEST_EFFORT
        return self._switch_cli.call_async(req)

    def _unblock_forward(self):
        self._block_forward = False
        self.get_logger().info("Drag off: command forwarding resumed")

    def _drag_cb(self, request, response):
        if request.data:
            if not self._gc_ok:
                response.success = False
                response.message = "gravity model not loaded; cannot enter drag"
                self.get_logger().warn(response.message)
                return response
            if not self.enable_flag:
                response.success = False
                response.message = "enable arm before drag"
                self.get_logger().warn(response.message)
                return response
            # JTC 명령 전달 먼저 차단 → arm_controller 비활성화 → 중력보상 MIT 시작
            self._block_forward = True
            self._switch_controller(activate=[], deactivate=[self.arm_controller_name])
            self.drag_mode_active = True
            response.success = True
            response.message = "Drag(gravity-comp) on"
            self.get_logger().info(response.message)
        else:
            # MIT 중단(발행 스레드 falling-edge가 현재 자세 hold/서보 이어받음) → arm_controller 재활성화
            self.drag_mode_active = False
            fut = self._switch_controller(activate=[self.arm_controller_name], deactivate=[])
            # 재활성화가 끝난 뒤에 전달 재개(전환 중 unclaimed 명령 인터페이스 stale 값 누출 방지)
            if fut is not None:
                fut.add_done_callback(lambda _f: self._unblock_forward())
            else:
                self._unblock_forward()   # 서비스 없으면 best-effort 즉시 해제
            response.success = True
            response.message = "Drag(gravity-comp) off (normal control restored)"
            self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NeroHardwareInterface()
    # control 타이머(전용 콜백그룹)와 command/drag 콜백을 다른 스레드에서 동시 실행
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
