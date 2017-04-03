from setuptools import setup, find_packages

setup(
    name='fwarp',
    version='0.0.16',
    description='Utility for warping functions',
    url='https://github.com/n-s-f/warp',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
)
