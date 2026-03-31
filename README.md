# fastslam_ros
This project implements FastSLAM, a particle filter-based SLAM algorithm that jointly estimates robot trajectory and map features.

Each particle maintains its own map, enabling scalable SLAM in complex environments.

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-39-35" src="https://github.com/user-attachments/assets/50858a37-80a4-4064-86d3-1a85c5812ab3" />

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-36-18" src="https://github.com/user-attachments/assets/5c73ec7b-99ae-4dd7-8f2f-00797b184c71" />


## Simulation Setup
All algorithms are implemented and tested using:
- ROS2 (Humble)
- Gazebo
- ArticubotOne robot (based on John Evan’s tutorial)
