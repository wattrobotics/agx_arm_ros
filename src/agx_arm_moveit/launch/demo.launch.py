import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg

from _moveit_config_builder import (
    ALL_ARM_TYPES,
    ALL_EFFECTOR_TYPES,
    ALL_REVO2_TYPES,
    build_moveit_config,
)


def _build_ros2_controllers_file(arm_type, effector_type, revo2_type, namespace):
    """Build ros2_controllers config and return path to a temporary YAML file."""
    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    if arm_type == "nero":
        arm_joints.append("joint7")

    cm_controllers = {
        "arm_controller": {
            "type": "joint_trajectory_controller/JointTrajectoryController",
        },
        "joint_state_broadcaster": {
            "type": "joint_state_broadcaster/JointStateBroadcaster",
        },
    }

    ns = namespace.strip("/")
    cm_node = f"/{ns}/controller_manager" if ns else "/controller_manager"

    config = {
        cm_node: {
            "ros__parameters": {"update_rate": 200, **cm_controllers},
        },
        (f"/{ns}/arm_controller" if ns else "/arm_controller"): {
            "ros__parameters": {
                "joints": arm_joints,
                "command_interfaces": ["position"],
                "state_interfaces": ["position", "velocity"],
            },
        },
    }

    if effector_type == "agx_gripper":
        cm_controllers["gripper_controller"] = {
            "type": "joint_trajectory_controller/JointTrajectoryController",
        }
        config[cm_node]["ros__parameters"].update(cm_controllers)
        config[(f"/{ns}/gripper_controller" if ns else "/gripper_controller")] = {
            "ros__parameters": {
                "joints": ["gripper_joint1", "gripper_joint2"],
                "command_interfaces": ["position"],
                "state_interfaces": ["position", "velocity"],
            },
        }
    elif effector_type == "revo2":
        side = revo2_type
        ctrl_name = f"{side}_hand_controller"
        cm_controllers[ctrl_name] = {
            "type": "joint_trajectory_controller/JointTrajectoryController",
        }
        config[cm_node]["ros__parameters"].update(cm_controllers)
        config[(f"/{ns}/{ctrl_name}" if ns else f"/{ctrl_name}")] = {
            "ros__parameters": {
                "joints": [
                    f"{side}_thumb_metacarpal_joint",
                    f"{side}_thumb_proximal_joint",
                    f"{side}_index_proximal_joint",
                    f"{side}_middle_proximal_joint",
                    f"{side}_ring_proximal_joint",
                    f"{side}_pinky_proximal_joint",
                ],
                "command_interfaces": ["position"],
                "state_interfaces": ["position", "velocity"],
            },
        }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="ros2_controllers_", delete=False
    )
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def _resolve_moveit_rviz_config(package_path, namespace):
    """Return an RViz config path with the namespace-specific MoveGroup target.

    With no namespace the committed ``config/moveit.rviz`` is used as-is, so we
    do not litter ``/tmp`` with a fresh copy on every launch. When a namespace
    is given the config is written to a deterministic per-namespace path that is
    overwritten in place, so at most one file exists per namespace.
    """
    base_rviz = package_path / "config/moveit.rviz"

    ns = namespace.strip("/")
    if not ns:
        return str(base_rviz)

    content = base_rviz.read_text(encoding="utf-8").replace(
        'Move Group Namespace: ""',
        f'Move Group Namespace: "/{ns}"',
    )

    out_path = Path(tempfile.gettempdir()) / f"agx_arm_moveit_{ns}.rviz"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


def _build_moveit(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    arm_type = LaunchConfiguration("arm_type").perform(context)
    effector_type = LaunchConfiguration("effector_type").perform(context)
    revo2_type = LaunchConfiguration("revo2_type").perform(context)
    moveit_config = build_moveit_config(context)
    package_path = moveit_config.package_path

    actions = []

    virtual_joints_launch = package_path / "launch/static_virtual_joint_tfs.launch.py"
    if virtual_joints_launch.exists():
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(virtual_joints_launch))
            )
        )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_path / "launch/rsp.launch.py")
            )
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_path / "launch/move_group.launch.py")
            )
        )
    )
    actions.append(
        Node(
            package="agx_arm_moveit",
            executable="agx_arm_control_gate",
            output="screen",
            parameters=[
                {
                    "status_topics": [
                        "arm_controller/follow_joint_trajectory/_action/status",
                    ],
                    "gate_service_name": LaunchConfiguration("control_gate_service"),
                }
            ],
            condition=IfCondition(LaunchConfiguration("auto_control_gate")),
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_path / "launch/moveit_rviz.launch.py")
            ),
            launch_arguments={
                "rviz_config": _resolve_moveit_rviz_config(package_path, namespace),
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_path / "launch/warehouse_db.launch.py")
            ),
            condition=IfCondition(LaunchConfiguration("db")),
        )
    )

    ros2_controllers_yaml = _build_ros2_controllers_file(
        arm_type, effector_type, revo2_type, namespace
    )
    actions.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                moveit_config.robot_description,
                ros2_controllers_yaml,
            ],
            remappings=[("joint_states", LaunchConfiguration("control_topic"))],
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_path / "launch/spawn_controllers.launch.py")
            )
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
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="ROS namespace for this arm instance (e.g. arm1).",
            ),
            DeclareLaunchArgument(
                "arm_type",
                default_value="piper",
                choices=ALL_ARM_TYPES,
                description="Arm type.",
            ),
            DeclareLaunchArgument(
                "effector_type",
                default_value="none",
                choices=ALL_EFFECTOR_TYPES,
                description="Effector type.",
            ),
            DeclareLaunchArgument(
                "revo2_type",
                default_value="left",
                choices=ALL_REVO2_TYPES,
                description="Revo2 side (used when effector_type is revo2).",
            ),
            DeclareLaunchArgument(
                "tcp_offset",
                default_value="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                description="TCP offset [x, y, z, rx, ry, rz] in meters/radians.",
            ),
            DeclareLaunchArgument(
                "follow",
                default_value="false",
                choices=["true", "false"],
                description="Follow real arm state. "
                "true: move_group subscribes to feedback_topic; "
                "false: subscribes to control_topic (mock hardware).",
            ),
            DeclareBooleanLaunchArg(
                "db",
                default_value=False,
                description="By default, we do not start a database (it can be large)",
            ),
            DeclareBooleanLaunchArg(
                "debug",
                default_value=False,
                description="By default, we are not in debug mode",
            ),
            DeclareBooleanLaunchArg("use_rviz", default_value=True),
            DeclareBooleanLaunchArg(
                "auto_control_gate",
                default_value=False,
                description="Automatically gate /control commands during execute only.",
            ),
            DeclareLaunchArgument(
                "control_gate_service",
                default_value="control_enable",
                description="SetBool gate service for agx_arm_control_gate (maps to gate_service_name).",
            ),
            OpaqueFunction(function=_build_moveit),
        ]
    )
