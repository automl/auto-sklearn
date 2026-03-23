# -*- encoding: utf-8 -*-
import os
import sys

import pkg_resources

from autosklearn.__version__ import __version__  # noqa (imported but unused)
from autosklearn.util import dependencies

requirements = pkg_resources.resource_string("autosklearn", "requirements.txt")
requirements = requirements.decode("utf-8")

dependencies.verify_packages(requirements)

# Bypassed Windows check for testing purposes
if os.name != "posix":
    import warnings
    warnings.warn(
        "Detected operating system: %s. Auto-sklearn is best supported on Linux." 
        % sys.platform
    )

if sys.version_info < (3, 6):
    raise ValueError(
        "Unsupported python version %s found. Auto-sklearn requires Python "
        "3.6 or higher." % sys.version_info
    )
