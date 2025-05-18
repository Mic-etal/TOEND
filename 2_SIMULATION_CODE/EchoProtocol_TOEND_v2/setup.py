# setup.py
from setuptools import setup, find_packages

setup(
    name="EchoProtocol",
    packages=find_packages(include=["EchoProtocol", "EchoProtocol.*"]),
    include_package_data=True
)