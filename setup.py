# -*- encoding: utf-8 -*-
import setuptools
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize
import os
import sys


# Check if Auto-sklearn *could* run on the given system
if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: http://automl.github.io'
        '/auto-sklearn/stable/installation.html#windows-osx-compability' %
        sys.platform
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python '
        '3.5 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )


extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])

requirements = [
    "setuptools",
    "nose",
    "Cython",
    # Numpy version of higher than 1.14.5 causes libgcc_s.so.1 error.
    "numpy>=1.9.0,<=1.14.5",
    "scipy>=0.14.1",
    "scikit-learn>=0.19,<0.20",
    "lockfile",
    "joblib",
    "psutil",
    "pyyaml",
    "liac-arff",
    "pandas",
    "ConfigSpace>=0.4.0,<0.5",
    "pynisher>=0.4.2",
    "pyrfr>=0.6.1,<0.8",
    "smac>=0.8,<0.9",
    "xgboost>=0.80",
]

with open("autosklearn/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=version,
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    test_suite='nose.collector',
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    python_requires='>=3.5.*',
    url='https://automl.github.io/auto-sklearn',
)
