# -*- encoding: utf-8 -*-
from distutils.extension import Extension
import subprocess

import setuptools
from setuptools.command.install import install
from pip.req import parse_requirements

from scripts import download_binaries
from autosklearn.conf import SCRIPT_COMPILE_C_UTILS


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
    version='0.0.2dev',
    ext_modules=[Extension('autosklearn.data.competition_c_functions',
                           ['autosklearn/data/competition_c_functions.c'])],
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=[str(ir.req) for ir in
                      parse_requirements('requirements.txt')],
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
