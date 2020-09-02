# -*- encoding: utf-8 -*-
import os
import pkg_resources
import sys

from autosklearn.util import dependencies
from autosklearn.__version__ import __version__  # noqa (imported but unused)


requirements = pkg_resources.resource_string('autosklearn', 'requirements.txt')
requirements = requirements.decode('utf-8')

dependencies.verify_packages(requirements)

if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: http://automl.github.io'
        '/auto-sklearn/stable/installation.html#windows-osx-compability' %
        sys.platform
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.5 or higher.' % sys.version_info
    )
