"""실물 Nero ↔ ros2_control 브리지(nero_hardware_interface)만 단독 실행.

MoveIt/ros2_control 과 인터페이스를 따로 켜고 싶을 때 사용한다. 0스냅 방지를 위해 이 인터페이스를
먼저 띄워 feedback/joint_states 가 흐르게 한 뒤, MoveIt 쪽을 start_interface:=false 로 실행한다.

실행: ros2 launch agx_arm_ctrl nero_interface.launch.py can_port:=can0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("can_port", default_value="can0"),
        DeclareLaunchArgument("arm_type", default_value="nero"),
        DeclareLaunchArgument("auto_enable", default_value="true"),
        DeclareLaunchArgument("speed_percent", default_value="100"),
        DeclareLaunchArgument("pub_rate", default_value="200"),
        DeclareLaunchArgument("enable_timeout", default_value="5.0"),
        Node(
            package="agx_arm_ctrl",
            executable="nero_hardware_interface",
            name="nero_hardware_interface",
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[{
                "can_port": LaunchConfiguration("can_port"),
                "arm_type": LaunchConfiguration("arm_type"),
                "auto_enable": LaunchConfiguration("auto_enable"),
                "speed_percent": LaunchConfiguration("speed_percent"),
                "pub_rate": LaunchConfiguration("pub_rate"),
                "enable_timeout": LaunchConfiguration("enable_timeout"),
            }],
        ),
    ])
