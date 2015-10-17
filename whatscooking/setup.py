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
    install_requires=[
        "Cython==0.23.4",
        "argparse==1.2.1",
        "boto==2.38.0",
        "bz2file==0.98",
        "docopt==0.6.2",
        "gensim==0.12.2",
        "httpretty==0.8.6",
        "numpy==1.10.1",
        "requests==2.8.1",
        "scikit-learn==0.16.1",
        "scipy==0.16.0",
        "six==1.10.0",
        "smart-open==1.3.0",
        "wsgiref==0.1.2"
    ],
    entry_points={
        'console_scripts': [
            'kaggle = whatscooking.whatscooking_wv:main',
        ]
    },

    # Magic !
    zip_safe=False
)
