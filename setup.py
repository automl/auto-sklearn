# -*- encoding: utf-8 -*-
import os

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

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

requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
requirements = []
dependency_links = []
with open(requirements_file) as fh:
    for line in fh:
        line = line.strip()
        if line:
            requirements.append(line)


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
