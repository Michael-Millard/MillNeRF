"""
Setup script for MillNeRF package.
"""

from setuptools import setup, find_packages

setup(
    name="millnerf",
    version="0.1.0",
    description="Neural Radiance Fields implementation from scratch",
    author="Michael Millard",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.0",
        "tqdm>=4.60.0",
        "PyYAML>=5.4.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        'console_scripts': [
            'millnerf=src.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)