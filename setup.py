# -*- encoding: utf-8 -*-
import os
import sys

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension

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

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

extensions = [Extension('autosklearn.data.competition_c_functions',
              sources=['autosklearn/data/competition_c_functions.pyx'])]

setup_requirements = [
    "Cython",
    "numpy"
]

requirements = [
    "setuptools",
    "nose",
    "six",
    "Cython",
    "numpy>=1.9.0",
    "scipy>=0.14.1",
    "scikit-learn>=0.19,<0.20",
    "lockfile",
    "joblib",
    "psutil",
    "pyyaml",
    "liac-arff",
    "pandas",
    "ConfigSpace>=0.4.0,<0.5",
    "pynisher>=0.4,<0.5",
    "pyrfr>=0.6.1,<0.8",
    "smac>=0.8,<0.9",
    "xgboost==0.7.post3",
]

with open("autosklearn/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=version,
    cmdclass={'build_ext': build_ext},
    setup_requires=setup_requirements,
    install_requires=requirements,
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
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
