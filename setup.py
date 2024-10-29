#!/usr/bin/env python3

import setuptools
from setuptools import setup

setup(
    name="VL - Visual Localization",
    version="0.0.1",
    author="2acholsk1",
    license="",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'VL = src.__main__:main'
        ],
    },
    python_requires='>=3.10',
)
