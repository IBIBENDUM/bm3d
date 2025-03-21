from setuptools import setup, find_packages

setup(
    name="BM3D",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.11.0.86",
        "PyWavelets>=1.8.0",
        "typing>=3.5",
        "scipy>=1.15.2"
    ],
)
