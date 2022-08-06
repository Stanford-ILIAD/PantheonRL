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
        'stable-baselines3',
        'scipy',
        'tqdm'
      ],
      )
