from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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
        default_value='nero',
        choices=['nero'],
        description='Robotic arm type (simple node is Nero/firmware v111 only).'
    )

    effector_type_arg = DeclareLaunchArgument(
        'effector_type',
        default_value='none',
        choices=['none', 'handeye'],
        description='End effector type. Only joint-less types are valid for the simple '
                    'node (handeye adds camera frames only; gripper/revo2 are unsupported).'
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
        description='Enable fast mode (move_js) for the AGX Arm node.'
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
        description='TCP offset in x, y, z, roll, pitch, yaw (used by MoveIt only).'
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

    # ── agx_arm_ctrl (simple node) ───────────────────────────────────
    agx_arm_node = Node(
        package='agx_arm_ctrl',
        executable='agx_arm_ctrl_simple',
        name='agx_arm_ctrl_simple_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'arm_type': LaunchConfiguration('arm_type'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'fast_mode': LaunchConfiguration('fast_mode'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'pub_rate': LaunchConfiguration('pub_rate'),
            'enable_timeout': LaunchConfiguration('enable_timeout'),
            'control_enabled': IfElseSubstitution(
                LaunchConfiguration('auto_control_gate'),
                if_value='false',
                else_value='true',
            ),
        }],
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
            'revo2_type': 'left',
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
        auto_enable_arg,
        fast_mode_arg,
        speed_percent_arg,
        pub_rate_arg,
        enable_timeout_arg,
        tcp_offset_arg,
        follow_arg,
        feedback_topic_arg,
        control_topic_arg,
        auto_control_gate_arg,
        control_gate_service_arg,
        # launches
        agx_arm_node,
        moveit_launch,
    ])
