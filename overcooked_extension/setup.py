#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='overcooked_ai',
      version='0.0.1',
      description='Cooperative multi-agent environment based on Overcooked',
      author='',
      author_email='',
      packages=find_packages(where="./src"),
      package_dir={'': 'src'},
      install_requires=[
        'numpy',
        'tqdm',
        'gym'
      ]
    )
