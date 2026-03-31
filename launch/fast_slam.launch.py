"""
fast_slam launch file
======================
Starts:
  1. fast_slam_node  — subscribes /scan + /odom, publishes:
                         /fast_slam/particles   PoseArray
                         /fast_slam/pose        PoseStamped
                         /fast_slam/landmarks   MarkerArray
                         /fast_slam/trajectory  Path
  2. rviz2           — visualises the particle cloud, landmark map,
                       best-particle trajectory, and laser scan

Usage:
  ros2 launch fast_slam fast_slam.launch.py

Override parameters at runtime:
  ros2 launch fast_slam fast_slam.launch.py \\
      params_file:=/path/to/my_params.yaml

Override RViz config:
  ros2 launch fast_slam fast_slam.launch.py \\
      rviz_config:=/path/to/my_config.rviz

The articubot simulation must be running beforehand so that /scan and
/odom are being published.  Start it with:
  ros2 launch articubot_one launch_sim.launch.py world:=./src/articubot_one/worlds/obstacles.world
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = FindPackageShare('fast_slam')

    # ── Launch arguments ──────────────────────────────────────────────────────
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg, 'config', 'fast_slam_params.yaml']),
        description='Full path to the FastSLAM parameters YAML file',
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg, 'rviz', 'fast_slam.rviz']),
        description='Full path to the RViz2 config file',
    )

    # ── 1. FastSLAM node ──────────────────────────────────────────────────────
    fast_slam_node = Node(
        package='fast_slam',
        executable='fast_slam_node',
        name='fast_slam_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        # /scan and /odom match the articubot topics directly — no remapping needed
    )

    # ── 2. RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
    )

    return LaunchDescription([
        params_arg,
        rviz_arg,
        fast_slam_node,
        rviz_node,
    ])
