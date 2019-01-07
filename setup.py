# -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


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
        'Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python '
        '3.5 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )


class BuildExt(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


extensions = [
    Extension('autosklearn.data.competition_c_functions',
              sources=['autosklearn/data/competition_c_functions.pyx'],
              language='c',
              extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
              extra_link_args=['-fopenmp'])
]

requirements = [
    "setuptools",
    "nose",
    "Cython",
    # Numpy version of higher than 1.14.5 causes libgcc_s.so.1 error.
    "numpy>=1.9.0,<=1.14.5",
    "scipy>=0.14.1",
    "scikit-learn>=0.19,<0.20",
    "lockfile",
    "joblib",
    "psutil",
    "pyyaml",
    "liac-arff",
    "pandas",
    "ConfigSpace>=0.4.0,<0.5",
    "pynisher>=0.4.2",
    "pyrfr>=0.6.1,<0.8",
    "smac>=0.8,<0.9",
    "xgboost>=0.80",
]

with open("autosklearn/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=version,
    cmdclass={'build_ext': BuildExt},
    ext_modules=extensions,
    packages=find_packages(exclude=['test', 'scripts', 'examples']),
    setup_requires=['numpy'],
    install_requires=requirements,
    test_suite='nose.collector',
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    python_requires='>=3.5.*',
    url='https://automl.github.io/auto-sklearn',
)
