from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fast_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aun',
    maintainer_email='you@example.com',
    description='FastSLAM 1.0 for ROS 2 Humble (Ex07)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fast_slam_node = fast_slam.fast_slam_node:main',
        ],
    },
)
