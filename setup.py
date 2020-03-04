#!/usr/bin/env python
from setuptools import setup


if __name__ == '__main__':

    setup(
        name="OpenVibSpec",
        version="0.0.1",
        package_dir={'': 'scr'},
        packages=['openvibspec'],
        include_package_data=True,
        install_requires=[
            'matplotlib',
            'keras',
            'numpy',
            'scipy',
            'scikit-learn',
            'h5py',
            'tensorflow',
        ],
    )
