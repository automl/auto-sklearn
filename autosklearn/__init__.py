# -*- encoding: utf-8 -*-
import os
import sys

from autosklearn.util import dependencies
from autosklearn.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
numpy>=1.9
scikit-learn>=0.19,<0.20
lockfile>=0.10
smac>=0.8,<0.9
pyrfr>=0.6.1,<0.8
ConfigSpace>=0.4.0,<0.5
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)

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