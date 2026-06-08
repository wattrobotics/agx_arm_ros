#!/usr/bin/env python3
# -*-coding:utf8-*-
"""MoveIt 제어 전용 최소 노드.

control/joint_states 구독 → move_j/move_js, feedback/joint_states 발행.
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


class AgxArmSimpleNode(Node):

    def __init__(self):
        super().__init__("agx_arm_ctrl_simple_node")

        self.declare_parameter("can_port", "can0")
        self.declare_parameter("arm_type", "nero")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("fast_mode", False)
        self.declare_parameter("speed_percent", 100)
        self.declare_parameter("pub_rate", 200)
        self.declare_parameter("enable_timeout", 5.0)
        self.declare_parameter("control_enabled", True)

        self.can_port = self.get_parameter("can_port").value
        self.arm_type = self.get_parameter("arm_type").value
        self.auto_enable = self.get_parameter("auto_enable").value
        self.fast_mode = self.get_parameter("fast_mode").value
        self.speed_percent = self.get_parameter("speed_percent").value
        self.pub_rate = self.get_parameter("pub_rate").value
        self.enable_timeout = self.get_parameter("enable_timeout").value
        self.control_enabled = self.get_parameter("control_enabled").value

        self.enable_flag = False
        self.control_ready = False

        self._init_arm()

        self.joint_states_pub = self.create_publisher(JointState, "feedback/joint_states", 1)
        self.create_subscription(JointState, "control/joint_states", self._control_cb, 1)
        self.create_service(SetBool, "enable_agx_arm", self._enable_cb)
        self.create_service(SetBool, "control_enable", self._gate_cb)

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
        if not self.enable_flag:
            self.get_logger().warn("Agx_arm is not enabled, cannot control")
            return False
        return self.control_enabled

    ### publisher thread
    def _publish_loop(self):
        period = 1.0 / self.pub_rate
        while rclpy.ok():
            try:
                self._publish_once()
            except Exception as e:
                self.get_logger().warn(f"publish error: {e}")
            time.sleep(period)

    def _publish_once(self):
        if not self.agx_arm.is_ok():
            return
        if not self.control_ready and self._arm_ready():
            self.control_ready = True
            self.get_logger().info("Agx_arm feedback is ready, control enabled")

        js = self.agx_arm.get_joint_angles()
        if js is None or js.hz <= 0:
            return
        msg = JointState()
        msg.header.stamp = self._to_ros_time(js.timestamp)
        msg.name = list(self.arm_joint_names)
        msg.position = list(js.msg)
        self.joint_states_pub.publish(msg)

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
