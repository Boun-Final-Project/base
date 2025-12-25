import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'efe_igdm'

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
            "start = efe_igdm.igdm:main",
            "clear_viz = efe_igdm.clear_visualization:main",
        ],
    },
)
