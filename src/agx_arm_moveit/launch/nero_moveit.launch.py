"""실물 Nero(+handeye) MoveIt 단일 실행 launch.

실행: ros2 launch agx_arm_moveit nero_moveit.launch.py can_port:=can0
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg

from _moveit_config_builder import build_moveit_config

os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"


def _build(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    moveit_config = build_moveit_config(context)
    package_path = moveit_config.package_path

    actions = []

    # --- MoveIt 쪽 (기존 sub-launch 재사용, 수정 없음) ---
    virtual_joints_launch = package_path / "launch/static_virtual_joint_tfs.launch.py"
    if virtual_joints_launch.exists():
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(virtual_joints_launch))
            )
        )
    # rsp / move_group / moveit_rviz 는 follow:=true 를 상속받아 feedback/joint_states(실측)를 구독한다.
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_path / "launch/rsp.launch.py"))
        )
    )
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_path / "launch/move_group.launch.py"))
        )
    )
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_path / "launch/moveit_rviz.launch.py")),
            launch_arguments={
                "rviz_config": str(package_path / "config/moveit.rviz"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    # --- 실물 브리지 (innfos_node 역할). feedback 먼저 흘러야 HW state가 실측이 되므로 먼저 띄운다. ---
    actions.append(
        Node(
            package="agx_arm_ctrl",
            executable="nero_hardware_interface",
            name="nero_hardware_interface",
            output="screen",
            parameters=[{
                "can_port": LaunchConfiguration("can_port"),
                "arm_type": LaunchConfiguration("arm_type"),
                "auto_enable": LaunchConfiguration("auto_enable"),
                "speed_percent": LaunchConfiguration("speed_percent"),
                "pub_rate": LaunchConfiguration("pub_rate"),
                "enable_timeout": LaunchConfiguration("enable_timeout"),
            }],
            condition=IfCondition(LaunchConfiguration("start_interface")),
        )
    )

    # --- ros2_control: topic 기반 HW. broadcaster는 /joint_states(실측)로 발행(리맵 없음). ---
    ros2_controllers_yaml = str(package_path / "config/ros2_controllers_nero.yaml")
    actions.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                ros2_controllers_yaml,
            ],
            # control/joint_states 는 JointStateTopicSystem 의 명령 채널이므로 broadcaster를 거기에
            # 리맵하면 충돌한다 → 리맵하지 않고 기본 /joint_states 로 둔다.
            remappings=[],
        )
    )

    # --- 컨트롤러 스폰 순서: broadcaster 먼저 → 그 다음 arm_controller.
    # broadcaster가 뜨는 사이 브리지 feedback이 HW state(실측)로 들어오므로, arm_controller(JTC)는
    # 활성화 시 실측 상태를 잡아 현재 자세에서 hold/시작한다(0스냅 방지).
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "controller_manager"],
        output="screen",
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "controller_manager"],
        output="screen",
    )
    actions.append(jsb_spawner)
    actions.append(
        RegisterEventHandler(
            OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner])
        )
    )

    return [
        GroupAction(
            actions=[
                PushRosNamespace(namespace),
                SetRemap(src="/robot_description", dst="robot_description"),
                *actions,
            ]
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        # 로봇/이펙터는 고정(실물 Nero + handeye)
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("arm_type", default_value="nero"),
        DeclareLaunchArgument("effector_type", default_value="handeye"),
        DeclareLaunchArgument("revo2_type", default_value="left"),
        DeclareLaunchArgument("tcp_offset", default_value="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"),
        # MoveIt/RSP가 실측(feedback/joint_states)을 구독하도록 (sub-launch들이 상속)
        DeclareLaunchArgument("follow", default_value="true"),
        DeclareLaunchArgument("feedback_topic", default_value="feedback/joint_states"),
        DeclareLaunchArgument("control_topic", default_value="control/joint_states"),
        # 브리지(nero_hardware_interface) 파라미터
        DeclareLaunchArgument("can_port", default_value="can0"),
        DeclareLaunchArgument("auto_enable", default_value="true"),
        DeclareLaunchArgument("speed_percent", default_value="100"),
        DeclareLaunchArgument("pub_rate", default_value="200"),
        DeclareLaunchArgument("enable_timeout", default_value="5.0"),
        # start_interface:=false 면 브리지(nero_hardware_interface)를 띄우지 않는다
        # (브리지를 nero_interface.launch.py 로 따로 띄울 때 사용).
        DeclareLaunchArgument("start_interface", default_value="true"),
        DeclareBooleanLaunchArg("use_rviz", default_value=True),
        OpaqueFunction(function=_build),
    ])
