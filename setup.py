# setup.py
from setuptools import setup, find_packages

setup(
    name='dynamic_neural_textures',
    version='0.1.0',
    description='A PyTorch implementation of dynamic neural textures',
    author='Your Name',
    author_email='your_email@example.com',
    url='https://github.com/Jac678/dynamic_neural_textures',
    packages=find_packages(),
    py_modules=[
        'physics_aware_upscaler',
        'optical_flow'  
    ],
    install_requires=[
        'torch>=1.8.0',
        'torchvision',
        'numpy',
        'matplotlib',
        'opencv-python',
        'pybullet',
        'warp-lang'
    ],
)

