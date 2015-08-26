# -*- encoding: utf-8 -*-
from distutils.core import setup

from Cython.Build import cythonize

setup(
    name='c_utils',
    version='0.1',
    ext_modules=cythonize('competition_c_functions.pyx'),
)
