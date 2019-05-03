""" This file is used to create the package. """
from setuptools import setup, find_packages

__version__ = '0.0.2'


setup(
    name='deeprl',
    version=__version__,
    packages=find_packages(include=['deeprl']),
    install_requires=[
        'numpy',
        'tensorflow',
        'click',
        'gym'
    ],
    author='Tom Henighan',
    entry_points={
        'console_scripts': [
            'deeprl = deeprl.manage:cli'
        ]
    }
)
