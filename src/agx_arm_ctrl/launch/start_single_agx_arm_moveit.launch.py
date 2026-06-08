from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
)
from launch.substitutions import LaunchConfiguration, IfElseSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

def generate_launch_description():

    # ── arguments ────────────────────────────────────────────────────
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal).'
    )
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='ROS namespace for this arm instance (e.g. arm1).'
    )

    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port to be used by the AGX Arm node.'
    )

    arm_type_arg = DeclareLaunchArgument(
        'arm_type',
        default_value='piper',
        choices=['nero', 'piper', 'piper_h', 'piper_l', 'piper_x'],
        description='Robotic arm type (e.g. nero, piper, piper_h, piper_l, piper_x).'
    )

    effector_type_arg = DeclareLaunchArgument(
        'effector_type',
        default_value='none',
        choices=['none', 'agx_gripper', 'revo2', 'handeye'],
        description='End effector type (e.g. agx_gripper, revo2, handeye).'
    )

    revo2_type_arg = DeclareLaunchArgument(
       'revo2_type',
        default_value='left',
        choices=['left', 'right'],
        description='Revo2 end effector type (e.g. left, right).'
    )

    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        choices=['true', 'false'],
        description='Automatically enable the AGX Arm node.'
    )

    fast_mode_arg = DeclareLaunchArgument(
        'fast_mode',
        default_value='false',
        choices=['true', 'false'],
        description='Enable fast mode for the AGX Arm node.'
    )

    speed_percent_arg = DeclareLaunchArgument(
        'speed_percent',
        default_value='100',
        description='Movement speed as a percentage of maximum speed.'
    )

    pub_rate_arg = DeclareLaunchArgument(
        'pub_rate',
        default_value='200',
        description='Publishing rate for the AGX Arm node.'
    )

    enable_timeout_arg = DeclareLaunchArgument(
        'enable_timeout',
        default_value='5.0',
        description='Timeout in seconds for arm enable/disable operations.'
    )

    tcp_offset_arg = DeclareLaunchArgument(
        'tcp_offset',
        default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
        # default_value='[0.0167, -0.016, -0.024, -0.785, 1.571, 2.356]',
        description='TCP offset in x, y, z, roll, pitch, yaw in meters/radians.'
    )

    gripper_default_effort_arg = DeclareLaunchArgument(
        'gripper_default_effort',
        default_value='1.0',
        description='Default effort for gripper commands (>= 0.0).'
    )

    follow_arg = DeclareLaunchArgument(
        'follow',
        default_value='true',
        choices=['true', 'false'],
        description='Follow real arm state.',
    )

    feedback_topic_arg = DeclareLaunchArgument(
        'feedback_topic',
        default_value='feedback/joint_states',
        description='Joint states feedback topic for MoveIt (follow:=true).',
    )

    control_topic_arg = DeclareLaunchArgument(
        'control_topic',
        default_value='control/joint_states',
        description='Joint states control topic for MoveIt (follow:=false, ros2_control remap).',
    )

    auto_control_gate_arg = DeclareLaunchArgument(
        'auto_control_gate',
        default_value='false',
        choices=['true', 'false'],
        description='Open control gate only during MoveIt execute stage.',
    )

    control_gate_service_arg = DeclareLaunchArgument(
        'control_gate_service',
        default_value='control_enable',
        description='SetBool gate service for agx_arm_control_gate when auto_control_gate:=true.',
    )

    collision_guard_enabled_arg = DeclareLaunchArgument(
        'collision_guard_enabled', default_value='true', choices=['true', 'false'],
        description='On collision (foc bit), auto switch to leader mode and recover on settle.',
    )
    crash_protection_level_arg = DeclareLaunchArgument(
        'crash_protection_level', default_value='[8, 8, 8, 8, 8, 8, 8]',
        description='Per-joint crash protection rating array (length=joint count, joint_index 1..7). '
                    'Each 0~8 (0=that joint off); all-zeros=off. Required for collision_guard.',
    )
    collision_window_sec_arg = DeclareLaunchArgument(
        'collision_window_sec', default_value='3.0',
        description='collision_guard: settle-judgement window (s).',
    )
    collision_std_deg_arg = DeclareLaunchArgument(
        'collision_std_deg', default_value='2.0',
        description='collision_guard: settle-judgement joint stddev threshold (deg).',
    )

    # ── agx_arm_ctrl ─────────────────────────────────────────────────
    agx_arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('agx_arm_ctrl'),
                'launch',
                'start_single_agx_arm.launch.py',
            )
        ),
        launch_arguments={
            'log_level': LaunchConfiguration('log_level'),
            'namespace': LaunchConfiguration('namespace'),
            'can_port': LaunchConfiguration('can_port'),
            'pub_rate': LaunchConfiguration('pub_rate'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'fast_mode': LaunchConfiguration('fast_mode'),
            'arm_type': LaunchConfiguration('arm_type'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'enable_timeout': LaunchConfiguration('enable_timeout'),
            'effector_type': LaunchConfiguration('effector_type'),
            'tcp_offset': LaunchConfiguration('tcp_offset'),
            'gripper_default_effort': LaunchConfiguration('gripper_default_effort'),
            'publish_gripper_joint': 'false',
            'control_enabled': IfElseSubstitution(
                LaunchConfiguration('auto_control_gate'),
                if_value='false',
                else_value='true',
            ),
            'collision_guard_enabled': LaunchConfiguration('collision_guard_enabled'),
            'crash_protection_level': LaunchConfiguration('crash_protection_level'),
            'collision_window_sec': LaunchConfiguration('collision_window_sec'),
            'collision_std_deg': LaunchConfiguration('collision_std_deg'),
        }.items(),
    )

    # ── agx_arm_moveit ───────────────────────────────────────────────
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('agx_arm_moveit'),
                'launch',
                'demo.launch.py',
            )
        ),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'arm_type': LaunchConfiguration('arm_type'),
            'effector_type': LaunchConfiguration('effector_type'),
            'revo2_type': LaunchConfiguration('revo2_type'),
            'tcp_offset': LaunchConfiguration('tcp_offset'),
            'follow': LaunchConfiguration('follow'),
            'feedback_topic': LaunchConfiguration('feedback_topic'),
            'control_topic': LaunchConfiguration('control_topic'),
            'auto_control_gate': LaunchConfiguration('auto_control_gate'),
            'control_gate_service': LaunchConfiguration('control_gate_service'),
        }.items(),
    )

    return LaunchDescription([
        # arguments
        log_level_arg,
        namespace_arg,
        can_port_arg,
        arm_type_arg,
        effector_type_arg,
        revo2_type_arg,
        auto_enable_arg,
        fast_mode_arg,
        speed_percent_arg,
        pub_rate_arg,
        enable_timeout_arg,
        tcp_offset_arg,
        gripper_default_effort_arg,
        follow_arg,
        feedback_topic_arg,
        control_topic_arg,
        auto_control_gate_arg,
        control_gate_service_arg,
        collision_guard_enabled_arg,
        crash_protection_level_arg,
        collision_window_sec_arg,
        collision_std_deg_arg,
        # launches
        agx_arm_launch,
        moveit_launch,
    ])
