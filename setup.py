#! /usr/bin/env python
# -*- coding: utf-8 -*-



from setuptools import setup, find_packages


setup(
    name="mlutils",
    version="0.1",
    description="machine learning utils",
    license="BSD",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

