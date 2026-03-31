# fastslam_ros
This project implements FastSLAM, a particle filter-based SLAM algorithm that jointly estimates robot trajectory and map features.

Each particle maintains its own map, enabling scalable SLAM in complex environments.



## Simulation Setup
All algorithms are implemented and tested using:
- ROS2 (Humble)
- Gazebo
- ArticubotOne robot (based on John Evan’s tutorial)
