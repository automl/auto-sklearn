# -*- encoding: utf-8 -*-

from __future__ import absolute_import
import subprocess

import setuptools

from setuptools import Extension
from setuptools.command.install import install

from pip.download import PipSession
from pip.req import parse_requirements

from conf import SCRIPT_INSTALL_REQS
from scripts import download_binaries

have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext

    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

C_UTILS_BASE = 'autosklearn/c_utils/competition_c_functions'
C_UTILS_LIB = 'autosklearn.c_utils.competition_c_functions'

class Download(install):

    def run(self):
        subprocess.call(["pip install -r requirements.txt --no-clean"], shell=True)
        download_binaries()
        subprocess.call(['bash', SCRIPT_INSTALL_REQS])
        install.do_egg_install(self)


if have_cython:
    c_utils = Extension(C_UTILS_LIB, ['%s.pyx' % C_UTILS_BASE])
else:
    c_utils = Extension(C_UTILS_LIB, ['%s.c' % C_UTILS_BASE])


setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.1dev',
    ext_modules=[c_utils],
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=[str(ir.req) for ir in
                      parse_requirements('requirements.txt',
                                         session=PipSession())],
    test_suite='nose.collector',
    cmdclass={'install': Download, 'build_ext': _build_ext},
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='www.automl.org')
