from setuptools import find_packages, setup

package_name = 'rrt_infotaxis'

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
            "start = rrt_infotaxis.rrt_infotaxis:main",
            'calibrate_q = rrt_infotaxis.calibrate_q:main',
            'calibrate_q_gaden = rrt_infotaxis.calibrate_q_gaden:main',
            'discrete_start = rrt_infotaxisBut.discrete_rrt_infotaxis:main',
        ],
    },
)
