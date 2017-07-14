# -*- encoding: utf-8 -*-
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
    "setuptools",
    "nose",
    "six",
    "Cython",
    "numpy>=1.9.0",
    "scipy>=0.14.1",
    "scikit-learn==0.18.1",
    "lockfile",
    "joblib",
    "psutil",
    "pyyaml",
    "liac-arff",
    "pandas",
    "ConfigSpace>=0.3.3,<0.4",
    "pynisher>=0.4",
    "pyrfr>=0.4,<0.5",
    "smac==0.5.0"
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
    url='https://automl.github.io/auto-sklearn')
