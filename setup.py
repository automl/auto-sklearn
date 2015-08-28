# -*- encoding: utf-8 -*-
import os
import shutil
from distutils.extension import Extension
import subprocess

import setuptools
from setuptools.command.install import install
from pip.req import parse_requirements

from scripts import download_binaries

install_reqs = parse_requirements('requirements.txt')

from autosklearn.conf import SCRIPT_COMPILE_C_UTILS


SMAC_DOWNLOAD_LOCATION = 'http://aad.informatik.uni-freiburg.de/~feurerm/'
SMAC_TAR_NAME = 'smac-v2.08.01-development-1.tar.gz'
# METADATA_LOCATION = "http://aad.informatik.uni-freiburg.de/~feurerm/"
# METADATA_TAR_NAME = "metadata_automl1_000.tar.gz"
RUNSOLVER_LOCATION = 'http://www.cril.univ-artois.fr/~roussel/runsolver/'
RUNSOLVER_TAR_NAME = 'runsolver-3.3.4.tar.bz2'
DOWNLOAD_DIRECTORY = os.path.join(os.path.dirname(__file__), '.downloads')
BINARIES_DIRECTORY = 'autosklearn/binaries'
METADATA_DIRECTORY = 'autosklearn/metalearning/files'


class Download(install):

    def run(self):
        download_binaries()

        subprocess.check_output([
            'bash',
            SCRIPT_COMPILE_C_UTILS
        ])
        install.do_egg_install(self)

setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.1dev',
    ext_modules=[Extension('autosklearn.data.competition_c_functions',
                           ['autosklearn/data/competition_c_functions.c'])],
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=[str(ir.req) for ir in install_reqs],
    test_suite='nose.collector',
    cmdclass={'install': Download},
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='www.automl.org')
