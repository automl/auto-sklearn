# -*- encoding: utf-8 -*-
import os
import shutil
import subprocess
import sys
import tarfile

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

import setuptools
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize
import autosklearn

extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])


setuptools.setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=autosklearn.__version__,
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=['numpy>=1.9.0',
                      'scipy>=0.14.1',
                      'scikit-learn==0.17.1',
                      'lockfile',
                      'psutil',
                      'pyyaml',
                      'six',
                      'ConfigArgParse',
                      'liac-arff',
                      'pandas',
                      'Cython',
                      'ConfigSpace',
                      'pynisher',
                      'smac',
                      'pyrfr'],
    test_suite='nose.collector',
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='automl.github.io/auto-sklearn')
