#!/usr/bin/env Python  
#coding=utf-8

from setuptools import setup
from setuptools import find_packages

setup(
    name='gym_gomoku',
    version='0.0.1',
    description = 'Game Gomoku or Five-In-a-Row Gymnasium Environment',
    author = 'Xichen Ding',
    author_email = 'dingo0927@126.com',
    url = 'https://github.com/rockingdingo/gym-gomoku',
    license="MIT",
    keywords='gymnasium gomoku reinforcement learning',
    packages=find_packages(),
    package_data={
        'demo': ['gym_gomoku_demo.gif', 
            'gym_gomoku_demo.json',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=['gymnasium>=0.26.0'],
    python_requires='>=3.7',
)
