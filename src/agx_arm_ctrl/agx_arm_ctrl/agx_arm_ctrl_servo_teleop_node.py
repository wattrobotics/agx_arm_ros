#!/usr/bin/env python3
# -*-coding:utf8-*-
"""범용 키보드 Twist teleop 노드 (하드웨어 비의존, latched 모델).

키보드로 6-DOF jog 명령을 받아 geometry_msgs/TwistStamped 를 파라미터로 지정한
토픽(기본 /servo_node/delta_twist_cmds)에 연속 발행한다. 기본 frame_id 는
end_point_link(엔드이펙터 좌표계)라서 MoveIt Servo 의 apply_twist_commands_about_ee_frame
와 합쳐지면 linear·angular 모두 EE 축 기준으로 jog 된다.

키입력 골격은 agx_arm_ctrl_teleop_with_move_p_node 에서 가져왔다(termios raw TTY +
select 논블로킹 + daemon thread + atexit 복원, 동일 JOG_KEYS).

** latched 모델 (teleop_twist_keyboard 식) **
raw 터미널에는 key-up 이벤트가 없어 "꾹 누르면 이동"은 OS 키 autorepeat 에 의존하는데,
첫 반복 지연(X11 기본 ~660ms)이 길어 깜빡인다. 그래서 이 노드는 키 반복에 의존하지 않는다:
  - jog 키를 한 번 누르면 그 축 속도가 "걸린다(latched)" → publish 타이머가 매 주기 그 속도를
    계속 발행 → 다른 키/정지키 전까지 유지.
  - SPACE 또는 k : 즉시 정지(twist 0).
  - 다른 jog 키 : 그 축으로 교체(단일 축만 활성, 예측 가능).
정지 시에도 zero twist 를 계속 발행해 Servo 를 TWIST 모드로 살려둔다(노드 종료 시 timeout 정지).

pyAgxArm/CAN/Pinocchio 의존이 전혀 없어 Servo 외의 임의 TwistStamped 소비 노드에도
재사용 가능하다(twist_topic / auto_switch_command_type 파라미터로 조정).

키보드 입력이 stdin TTY 를 필요로 하므로 launch 가 아니라 `ros2 run` 으로 실행할 것.
"""
import sys
import time
import select
import termios
import tty
import atexit
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

# 키 → (축 index 0..5, 부호). 0:x 1:y 2:z 3:roll 4:pitch 5:yaw
JOG_KEYS = {
    "w": (0, +1), "s": (0, -1),   # x
    "a": (1, +1), "d": (1, -1),   # y
    "r": (2, +1), "f": (2, -1),   # z
    "u": (3, +1), "o": (3, -1),   # roll
    "i": (4, +1), "k": (4, -1),   # pitch
    "j": (5, +1), "l": (5, -1),   # yaw
}

STOP_KEYS = {" "}                 # 정지(SPACE)

KEYMAP_BANNER = """\
========== servo twist teleop (latched) ==========
 linear  :  x+ w / x- s    y+ a / y- d    z+ r / z- f
 angular :  roll+ u / roll- o   pitch+ i / pitch- k   yaw+ j / yaw- l
 SPACE : 정지(latch 해제)
 c     : frame 토글 (EE end_point_link <-> base_link)
 [ / ] : linear_scale  down / up        - / = : angular_scale  down / up
 p     : 현재 frame / scale / 활성축 출력
 ESC / Ctrl-C : 종료
 * 키를 한 번 누르면 그 방향으로 계속 이동, 정지키 전까지 유지(latched) *
=================================================="""

SCALE_MIN = 0.001
SCALE_MAX = 2.0


class AgxArmServoTeleopNode(Node):

    def __init__(self):
        super().__init__("agx_arm_ctrl_servo_teleop")

        self.declare_parameter("twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("frame_id", "end_point_link")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_rate", 50.0)
        # unitless twist [-1,1]. servo가 scale.linear(0.2 m/s)·scale.rotational(0.6 rad/s)를 곱함.
        # 기본 0.005/0.01 → 실제 ≈ 0.001 m/s, 0.006 rad/s (미세 jog). [ ]/- = 로 런타임 조절.
        self.declare_parameter("linear_scale", 0.005)
        self.declare_parameter("angular_scale", 0.01)
        self.declare_parameter("scale_step", 1.5)
        self.declare_parameter("auto_switch_command_type", True)
        self.declare_parameter("command_type_service", "/servo_node/switch_command_type")
        self.declare_parameter("command_type_value", 1)   # 1 = TWIST

        self.twist_topic = self.get_parameter("twist_topic").value
        self.ee_frame = self.get_parameter("frame_id").value
        self.base_frame = self.get_parameter("base_frame_id").value
        self.publish_rate = max(1.0, float(self.get_parameter("publish_rate").value))
        self.linear_scale = float(self.get_parameter("linear_scale").value)
        self.angular_scale = float(self.get_parameter("angular_scale").value)
        self.scale_step = max(1.0001, float(self.get_parameter("scale_step").value))
        self.auto_switch = bool(self.get_parameter("auto_switch_command_type").value)
        self.command_type_service = self.get_parameter("command_type_service").value
        self.command_type_value = int(self.get_parameter("command_type_value").value)

        # latched 상태: active = (axis, sign) 또는 None(정지). twist 는 active+scale 로부터 산출.
        self.active = None
        self.twist = [0.0] * 6
        self.current_frame = self.ee_frame
        self._lock = threading.Lock()

        self.pub = self.create_publisher(TwistStamped, self.twist_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self._publish)

        # 시작 시 Servo 를 TWIST 모드로 전환(옵션). 서비스가 떠 있어야 하므로 1회성 타이머로 시도.
        self._switch_client = None
        if self.auto_switch:
            self._switch_timer = self.create_timer(0.5, self._try_switch_command_type)

        self.kbd_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.kbd_thread.start()

        self.get_logger().info(KEYMAP_BANNER)
        self.get_logger().info(
            f"twist_topic={self.twist_topic}  frame={self.current_frame}  "
            f"rate={self.publish_rate}Hz  lin={self.linear_scale}  ang={self.angular_scale}"
        )

    ### switch_command_type (Servo 편의 — 옵션)
    def _try_switch_command_type(self):
        """Servo 의 switch_command_type 서비스를 1회 호출해 TWIST 모드로 전환.
        서비스가 아직 없으면 다음 타이머 주기에 재시도, 성공/요청 후 타이머 종료."""
        from moveit_msgs.srv import ServoCommandType
        if self._switch_client is None:
            self._switch_client = self.create_client(ServoCommandType, self.command_type_service)
        if not self._switch_client.service_is_ready():
            return                                  # 다음 주기 재시도
        req = ServoCommandType.Request()
        req.command_type = self.command_type_value
        future = self._switch_client.call_async(req)
        future.add_done_callback(self._switch_done)
        self._switch_timer.cancel()

    def _switch_done(self, future):
        try:
            ok = future.result().success
        except Exception as e:                      # noqa: BLE001 - 서비스 실패는 치명적 아님
            self.get_logger().warn(f"switch_command_type 호출 실패: {e}")
            return
        self.get_logger().info(
            f"switch_command_type(TWIST) {'성공' if ok else '거부됨'}"
        )

    ### latched twist 산출
    def _rebuild_twist(self):
        """active(axis, sign) + 현재 scale 로부터 twist 6-vector 재구성. _lock 안에서 호출."""
        self.twist = [0.0] * 6
        if self.active is not None:
            axis, sign = self.active
            scale = self.linear_scale if axis < 3 else self.angular_scale
            self.twist[axis] = sign * scale

    ### publish loop
    def _publish(self):
        with self._lock:
            tw = list(self.twist)
            frame = self.current_frame
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = tw[0], tw[1], tw[2]
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = tw[3], tw[4], tw[5]
        self.pub.publish(msg)

    ### keyboard (teleop_with_move_p 골격 재사용)
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
                if ch == "\x03" or ch == "\x1b":    # Ctrl-C / ESC
                    self.get_logger().info("quit key received")
                    self._stop_and_shutdown()
                    break
                self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, ch):
        if ch in STOP_KEYS:
            with self._lock:
                self.active = None
                self._rebuild_twist()
            self.get_logger().info("STOP")
        elif ch in JOG_KEYS:
            axis, sign = JOG_KEYS[ch]
            with self._lock:
                self.active = (axis, sign)          # 단일 축 latched (교체)
                self._rebuild_twist()
                tw = list(self.twist)
            self.get_logger().info(
                "move axis%d sign%+d -> twist=[%s]" % (axis, sign,
                    ", ".join(f"{v:+.3f}" for v in tw))
            )
        elif ch == "c":
            with self._lock:
                self.current_frame = (
                    self.base_frame if self.current_frame == self.ee_frame else self.ee_frame
                )
                frame = self.current_frame
            self.get_logger().info(f"frame -> {frame}")
        elif ch == "[":
            with self._lock:
                self.linear_scale = max(SCALE_MIN, self.linear_scale / self.scale_step)
                self._rebuild_twist()
            self.get_logger().info(f"linear_scale={self.linear_scale:.3f}")
        elif ch == "]":
            with self._lock:
                self.linear_scale = min(SCALE_MAX, self.linear_scale * self.scale_step)
                self._rebuild_twist()
            self.get_logger().info(f"linear_scale={self.linear_scale:.3f}")
        elif ch == "-":
            with self._lock:
                self.angular_scale = max(SCALE_MIN, self.angular_scale / self.scale_step)
                self._rebuild_twist()
            self.get_logger().info(f"angular_scale={self.angular_scale:.3f}")
        elif ch == "=":
            with self._lock:
                self.angular_scale = min(SCALE_MAX, self.angular_scale * self.scale_step)
                self._rebuild_twist()
            self.get_logger().info(f"angular_scale={self.angular_scale:.3f}")
        elif ch == "p":
            with self._lock:
                tw = "[" + ", ".join(f"{v:+.3f}" for v in self.twist) + "]"
                frame = self.current_frame
                active = self.active
            self.get_logger().info(
                f"frame={frame}  lin={self.linear_scale:.3f}  ang={self.angular_scale:.3f}  "
                f"active={active}  twist={tw}"
            )

    def _stop_and_shutdown(self):
        """종료 전 zero twist 를 몇 번 발행해 Servo 가 정지하도록 한 뒤 shutdown."""
        with self._lock:
            self.active = None
            self._rebuild_twist()
        for _ in range(3):
            self._publish()
            time.sleep(0.02)
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(AgxArmServoTeleopNode())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
