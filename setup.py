# -*- encoding: utf-8 -*-
import autosklearn

import setuptools
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])

requirements = [
    "unittest2",
    "setuptools",
    "mock",
    "nose",
    "six",
    "Cython",
    "numpy>=1.9.0",
    "scipy>=0.14.1",
    "scikit-learn==0.17.1",
    "lockfile",
    "joblib",
    "psutil",
    "pyyaml",
    "ConfigArgParse",
    "liac-arff",
    "pandas",
    "xgboost==0.4a30",
    "ConfigSpace",
    "pynisher>=0.4",
    "pyrfr",
    "smac==0.0.1"
]


setuptools.setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=autosklearn.__version__,
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    test_suite='nose.collector',
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='https://automl.github.io/auto-sklearn')
