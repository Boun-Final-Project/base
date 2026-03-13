import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'igdm_multiple'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='efe',
    maintainer_email='efemantaroglu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "start = igdm_multiple.igdm:main",
            "start_basic = igdm_multiple.igdm_basic:main",
            "start_multi = igdm_multiple.igdm_multi:main",
            "clear_viz = igdm_multiple.tools.clear_visualization:main",
            "slam_node = igdm_multiple.slam_node:main",
        ],
    },
)
