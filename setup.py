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
        "tensorboard==2.14.0",
        "setuptools==59.5.0",
        "numpy>=1.16.4",
        "numpy<1.20.0",
        "GitPython",
        "onnx",
        'mujoco==3.2.3',
        'protobuf==3.20.3'
    ],
)