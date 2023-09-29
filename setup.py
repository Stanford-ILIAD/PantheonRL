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
        'flask==2.2.2',
        'tensorflow==2.11.0',
        'torch==1.13.1',
        'tensorboard==2.11.0',
        'stable-baselines3==1.7.0',
        'scipy==1.7.3',
        'tqdm==4.64.1'
      ],
      )
