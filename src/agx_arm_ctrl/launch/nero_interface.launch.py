"""실물 Nero ↔ ros2_control 브리지(nero_hardware_interface)만 단독 실행.

MoveIt/ros2_control 과 인터페이스를 따로 켜고 싶을 때 사용한다. 0스냅 방지를 위해 이 인터페이스를
먼저 띄워 feedback/joint_states 가 흐르게 한 뒤, MoveIt 쪽을 start_interface:=false 로 실행한다.

게인/가드/제어 튜닝 파라미터는 config/nero_hardware_interface.yaml(params_file)로 관리한다.
환경/구조 파라미터(can_port/arm_type/auto_enable)만 launch 인자로 넘긴다.

실행: ros2 launch agx_arm_ctrl nero_interface.launch.py can_port:=can0
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("agx_arm_ctrl"),
        "config", "nero_hardware_interface.yaml",
    )
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("can_port", default_value="can0"),
        DeclareLaunchArgument("arm_type", default_value="nero"),
        DeclareLaunchArgument("auto_enable", default_value="true"),
        # 튜닝 파라미터(게인/가드/pub_rate 등)는 이 yaml로. 커스텀 yaml로 교체 가능.
        DeclareLaunchArgument("params_file", default_value=default_params),
        Node(
            package="agx_arm_ctrl",
            executable="nero_hardware_interface",
            name="nero_hardware_interface",
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),          # 튜닝(YAML)
                {                                            # 환경/구조만
                    "can_port": LaunchConfiguration("can_port"),
                    "arm_type": LaunchConfiguration("arm_type"),
                    "auto_enable": LaunchConfiguration("auto_enable"),
                },
            ],
        ),
    ])
