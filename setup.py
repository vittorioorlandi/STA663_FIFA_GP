#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:27:51 2021

@author: grahamtierney
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'pybind11',
    'numpy',
    'pandas',
    'cppimport'
    ]

setuptools.setup(
    name="fifa-gp", 
    version="0.0.12",
    author="Alessandro Zito, Graham Tierney, and Vittorio Orlandi",
    author_email="vittorio.orlandi@duke.edu",
    description="Python implementation of FIFA-GP from Moran and Wheeler (2020)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vittorioorlandi/STA663_FIFA_GP",
    project_urls={
        "Bug Tracker": "https://github.com/vittorioorlandi/STA663_FIFA_GP/issues",
    },
    license = "LICENSE.txt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["fifa_gp"],
    include_package_data=True,
    install_requires = install_requires,
    python_requires=">=3.6",
)