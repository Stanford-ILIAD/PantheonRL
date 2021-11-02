#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pantheonrl',
      version='0.0.1',
      description='PantheonRL',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
        'flask',
        'tensorflow',
        'torch',
        'tensorboard',
        'stable-baselines3 @ https://github.com/DLR-RM/stable-baselines3/archive/2fa06ae8d2.zip#egg=stable-baselines3-1.2.0a0',
        'scipy',
        'tqdm'
      ],
      )
