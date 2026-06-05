import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'irob_lerobot_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + [
        (os.path.join('share', package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join(package_name, '**', '*'), recursive=True) if os.path.isfile(f)
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andras Makany',
    maintainer_email='makany.andras@uni-obuda.hu',
    description='LeRobot ROS 2 integration package.',
    license='GPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
