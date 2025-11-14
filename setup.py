from setuptools import find_packages
from distutils.core import setup

setup(
    name="loco-manipulation",
    version="1.0.0",
    author="kai",
    license="MIT License",
    packages=find_packages(),
    description="Isaac Gym environments for Loco-Manipulation Task",
    install_requires=[
        "isaacgym",
        "rsl_rl",
        "matplotlib",
        "tensorboard",
        "numpy==1.22.0",
        'mujoco==3.2.3',
    ],
)