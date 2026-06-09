#!/usr/bin/env python3
# -*-coding:utf8-*-
"""Nero 실물 팔 ↔ ros2_control(topic 기반 하드웨어 인터페이스) 브리지.

ROS1 `innfos_node` 역할의 최소 노드(1단계: drag/guard 없음).
  - command_topic(JTC 명령, topic HW의 write 발행) 구독 → Nero `move_js`
  - Nero 실측 → feedback_topic 발행 (topic HW의 read 구독 대상)
대응: innfos/input ↔ command_topic, innfos/output ↔ feedback_topic.

메시지 타입은 양 토픽 모두 sensor_msgs/JointState (topic_based_ros2_control/TopicBasedSystem 및
joint_state_topic_hardware_interface/JointStateTopicSystem 공통). 토픽 이름은 ros2_control xacro의
joint_commands_topic / joint_states_topic 과 반드시 일치시켜야 한다.
"""
import time
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW


class NeroHardwareInterface(Node):

    def __init__(self):
        super().__init__("nero_hardware_interface")

        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "nero")
        self.declare_parameter("auto_enable", True)
        # JTC position 셋포인트 스트리밍에는 move_js(고속)가 적합
        self.declare_parameter("fast_mode", True)
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 200)
        self.declare_parameter("enable_timeout", 5.0)
        # ros2_control xacro의 joint_commands_topic / joint_states_topic 과 일치해야 함
        self.declare_parameter("command_topic", "control/joint_states")
        self.declare_parameter("feedback_topic", "feedback/joint_states")

        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.command_topic = self.get_parameter("command_topic").value
        self.feedback_topic = self.get_parameter("feedback_topic").value

        self.enable_flag = False
        # 실측을 1회 이상 수신해 발행을 시작했는지(= ROS1 innfos position_set 가드 역할).
        # 이 전에는 명령을 Nero로 전달하지 않는다.
        self.control_ready = False

        self._init_arm()

        self.joint_states_pub = self.create_publisher(JointState, self.feedback_topic, 1)
        self.create_subscription(JointState, self.command_topic, self._command_cb, 1)
        self.create_service(SetBool, "enable_agx_arm", self._enable_cb)

        # 발행 전용 스레드(블로킹 CAN 호출을 executor와 분리; simple_node와 동일 패턴)
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()

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

    ### state 발행 (feedback_topic) → topic HW read → joint_state_broadcaster → /joint_states
    def _publish_loop(self):
        while rclpy.ok():
            try:
                self._publish_once()
            except Exception as e:
                self.get_logger().warn(f"publish error: {e}")
            time.sleep(max(0.0, 1.0 / self.pub_rate))

    def _publish_once(self):
        if not self.agx_arm.is_ok():
            return
        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            return
        if not self.control_ready:
            self.control_ready = True
            self.get_logger().info("Nero feedback is ready")

        q = list(js.msg)
        velocities, efforts = [], []
        for j in range(1, self.arm_joint_count + 1):
            ms = self.agx_arm.get_motor_states(j)
            velocities.append(ms.msg.velocity if ms is not None else 0.0)
            efforts.append(ms.msg.torque if ms is not None else 0.0)

        msg = JointState()
        msg.header.stamp = self._to_ros_time(js.timestamp)
        msg.name = list(self.arm_joint_names)
        msg.position = q
        msg.velocity = velocities
        msg.effort = efforts
        self.joint_states_pub.publish(msg)

    ### command 수신 (command_topic) → Nero
    def _command_cb(self, msg: JointState):
        # 실측 확정 + enable 전에는 명령을 무시(시작 시 0/초기치 전달 방지)
        if not (self.control_ready and self.enable_flag):
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

        if self.fast_mode:
            self.agx_arm.move_js(joints)
        else:
            self.agx_arm.move_j(joints)

    ### service
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


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(NeroHardwareInterface())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
