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
from setuptools.command.install import install
import numpy as np
from Cython.Build import cythonize

METADATA_DIRECTORY = 'autosklearn/metalearning/files'

extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])

requirements_file = os.path.join(os.path.dirname(__file__), 'requ.txt')
requirements = []
dependency_links = []
with open(requirements_file) as fh:
    for line in fh:
        line = line.strip()
        if line:
            # Make sure the github URLs work here as well
            split = line.split('@')
            split = split[0]
            split = split.split('/')
            url = '/'.join(split[:-1])
            requirement = split[-1]
            requirements.append(requirement)
            # Add the rest of the URL to the dependency links to allow
            # setup.py test to work
            if 'git+https' in url:
                dependency_links.append(line.replace('git+', ''))


setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.1dev',
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    dependency_links=dependency_links,
    test_suite='nose.collector',
    #cmdclass={'install': Download},
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='www.automl.org')
