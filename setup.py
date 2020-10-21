# -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages


# Check if Auto-sklearn *could* run on the given system
if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: https://automl.github.io'
        '/auto-sklearn/master/installation.html#windows-osx-compatibility' %
        sys.platform
    )

if sys.version_info < (3, 6):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python '
        '3.6 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

HERE = os.path.abspath(os.path.dirname(__file__))
setup_reqs = ['Cython', 'numpy']
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

extras_reqs={
    "test": [
        "pytest>=4.6",
        "mypy",
        "pytest-xdist",
        "pytest-timeout",
        "flaky",
        "pytest-cov",
        "openml",
    ]
}

with open("autosklearn/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


with open('README.md') as fh:
    long_description = fh.read()


setup(
    name='auto-sklearn',
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    description='Automated machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=version,
    packages=find_packages(exclude=['test', 'scripts', 'examples']),
    setup_requires=setup_reqs,
    extras_require=extras_reqs,
    install_requires=install_reqs,
    include_package_data=True,
    license='BSD',
    platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    url='https://automl.github.io/auto-sklearn',
)
