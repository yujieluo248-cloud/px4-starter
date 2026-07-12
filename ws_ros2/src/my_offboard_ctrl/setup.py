from setuptools import find_packages, setup

package_name = 'my_offboard_ctrl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cre',
    maintainer_email='cre@todo.todo',
    description='PX4 ROS 2 offboard control examples',
    license='BSD 3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_ctrl_example = my_offboard_ctrl.offboard_ctrl_example:main',
            'offboard_takeoff_hover = my_offboard_ctrl.offboard_takeoff_hover:main',
            'offboard_single_waypoint = my_offboard_ctrl.offboard_single_waypoint:main',
            'offboard_waypoint_patrol = my_offboard_ctrl.offboard_waypoint_patrol:main',
            'offboard_obstacle_aware_patrol = my_offboard_ctrl.offboard_obstacle_aware_patrol:main',
            'offboard_gate_alignment = my_offboard_ctrl.offboard_gate_alignment:main',
            'offboard_gate_three_point_calibration = my_offboard_ctrl.offboard_gate_three_point_calibration:main',
            'offboard_gate_vertical_calibration = my_offboard_ctrl.offboard_gate_vertical_calibration:main',
            'offboard_gate_step_approach = my_offboard_ctrl.offboard_gate_step_approach:main',
            'offboard_gate_height_micro_calibration = my_offboard_ctrl.offboard_gate_height_micro_calibration:main',
            'offboard_left_gate_single_traversal = my_offboard_ctrl.offboard_left_gate_single_traversal:main',
            'offboard_left_gate_single_traversal_corrected = my_offboard_ctrl.offboard_left_gate_single_traversal_corrected:main',
            'offboard_local_axis_calibration = my_offboard_ctrl.offboard_local_axis_calibration:main',
            'offboard_local_x_only_calibration = my_offboard_ctrl.offboard_local_x_only_calibration:main',
            'offboard_wide_gate_traversal_v1 = my_offboard_ctrl.offboard_wide_gate_traversal_v1:main',
            'offboard_wide_gate_traversal_v2 = my_offboard_ctrl.offboard_wide_gate_traversal_v2:main',
            'offboard_wide_gate_traversal_v3 = my_offboard_ctrl.offboard_wide_gate_traversal_v3:main',
            'offboard_wide_gate_pre_gate_check = my_offboard_ctrl.offboard_wide_gate_pre_gate_check:main',
        ],
    },
)
