#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
try:
    from setuptools import setup, find_packages
    from setuptools.command.test import test as TestCommand
except ImportError:
    from distutils.core import setup


VERSION = "0.0.1"


setup(
    # Project informations
    name='whatscooking',
    version=VERSION,
    author=u'remusao',
    author_email=u'remi.berson@gmail.com',

    # License and description
    license="DBAD",
    description=u"Source code for What's Cooking Kaggle",
    # Package, scripts informations
    packages=find_packages(exclude=["data"]),
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python :: 2.7',
        'License :: Other/Proprietary License'
    ],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'kaggle = whatscooking.whatscooking_wv:main',
        ]
    },

    # Magic !
    zip_safe=False
)
