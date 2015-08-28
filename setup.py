# -*- encoding: utf-8 -*-

from __future__ import absolute_import
import subprocess

from pip.download import PipSession
import setuptools
from setuptools.command.install import install
from pip.req import parse_requirements

from conf import SCRIPT_INSTALL_REQS
from scripts import download_binaries


class Download(install):

    def run(self):
        subprocess.call(["pip install -r requirements.txt --no-clean"], shell=True)
        download_binaries()
        subprocess.call(['bash', SCRIPT_INSTALL_REQS])

        from Cython.Build import cythonize

        ext = cythonize('autosklearn/c_utils/competition_c_functions.pyx')[0]
        ext.name = 'c_utils'
        ext.version = '0.1'
        self.ext_modules = [ext]
        install.do_egg_install(self)


setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.2dev',
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
