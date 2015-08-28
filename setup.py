# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from distutils.extension import Extension
import subprocess

from pip.download import PipSession

import setuptools
from setuptools.command.install import install

from pip.req import parse_requirements

from conf import SCRIPT_COMPILE_C_UTILS, SCRIPT_INSTALL_REQS
from scripts import download_binaries

class Download(install):

    def run(self):
        subprocess.call(["pip install -r requirements.txt --no-clean"], shell=True)
        download_binaries()
        subprocess.call(['bash', SCRIPT_INSTALL_REQS])

        subprocess.call(['bash', SCRIPT_COMPILE_C_UTILS])
        install.do_egg_install(self)

setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.2dev',
    ext_modules=[Extension('autosklearn.c_utils.competition_c_functions',
                           ['autosklearn/c_utils/competition_c_functions.c'])],
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=[str(ir.req) for ir in
                      parse_requirements('requirements.txt',
                                         session=PipSession())],
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

# subprocess.call(['bash %s' % SCRIPT_COMPILE_C_UTILS])
# subprocess.call(['bash', SCRIPT_COMPILE_C_UTILS])