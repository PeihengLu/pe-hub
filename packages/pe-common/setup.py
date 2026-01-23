"""Setup script for PE Common package"""
from setuptools import setup, find_packages

setup(
    name="pe-common",
    version="0.1.0",
    description="Shared utilities for PE Database and PE Ensemble services",
    author="PE DB Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "biopython>=1.79",
        "torch>=1.9.0",
        "tensorflow>=2.14.0",
        "viennarna>=2.5.0",  # For RNA structure prediction (MFE)
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
