from setuptools import find_packages, setup

package_name = 'infotaxis'

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
    maintainer_email='efe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'infotaxis_node = infotaxis.infotaxis_node:main',
            'keyboard_control = infotaxis.keyboard_control:main',
        ],
    },
)
