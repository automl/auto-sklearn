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
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.5 or higher.' % sys.version_info
    )


extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])

with open("requirements.txt") as reqs:
    requirements = reqs.read().replace("\r", "").split("\n")

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
    python_requires='>=3.4.*',
    url='https://automl.github.io/auto-sklearn',
)
