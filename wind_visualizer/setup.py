from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wind_visualizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='efe',
    maintainer_email='efemantaroglu@gmail.com',
    description='Wind field visualization package for GADEN environment',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'wind_viz_node = wind_visualizer.wind_viz_node:main'
        ],
    },
)
