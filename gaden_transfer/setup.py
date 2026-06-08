import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'gaden_transfer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch_lidar'),
            glob('gaden_transfer_lidar/launch/*.py')),
        (os.path.join('share', package_name, 'launch_lidar'),
            glob('gaden_transfer_lidar/launch/*.yaml')),
        (os.path.join('share', package_name, 'launch_image_5ch'),
            glob('gaden_transfer_image_5ch/launch/*.py')),
        (os.path.join('share', package_name, 'launch_image_5ch'),
            glob('gaden_transfer_image_5ch/launch/*.yaml')),
        (os.path.join('share', package_name, 'launch_image_6ch'),
            glob('gaden_transfer_image_6ch/launch/*.py')),
        (os.path.join('share', package_name, 'launch_image_6ch'),
            glob('gaden_transfer_image_6ch/launch/*.yaml')),
        (os.path.join('share', package_name, 'launch_nav'),
            glob('gaden_transfer_nav/launch/*.py')),
        (os.path.join('share', package_name, 'launch_nav'),
            glob('gaden_transfer_nav/launch/*.yaml')),
        (os.path.join('share', package_name, 'environment'),
            ['ament_hooks/gaden_transfer.sh.in']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='efe',
    maintainer_email='efemantaroglu@gmail.com',
    description='GADEN-side ROS deployment of PPO checkpoints trained in reinforcement_learning (lidar / image_5ch / image_6ch / nav variants).',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'gaden_rl_node_lidar = gaden_transfer.gaden_transfer_lidar.gaden_rl_node:main',
            'gaden_rl_node_image_5ch = gaden_transfer.gaden_transfer_image_5ch.gaden_rl_node:main',
            'gaden_rl_node_image_6ch = gaden_transfer.gaden_transfer_image_6ch.gaden_rl_node:main',
            'gaden_rl_node_nav = gaden_transfer.gaden_transfer_nav.gaden_rl_node:main',
        ],
    },
)
