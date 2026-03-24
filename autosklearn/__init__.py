# -*- encoding: utf-8 -*-
import os
import sys

import pkg_resources

from autosklearn.__version__ import __version__  # noqa (imported but unused)
from autosklearn.util import dependencies

try:
    requirements = pkg_resources.resource_string("autosklearn", "requirements.txt")
    requirements = requirements.decode("utf-8")
    dependencies.verify_packages(requirements)
except Exception as e:
    import warnings
    warnings.warn(f"Dependency verification skipped: {e}")

if os.name != "posix":
    import warnings
    warnings.warn(
        "Detected non-POSIX operating system: %s. Some features may not work correctly. "
        "Please check the compability information of auto-sklearn: https://automl.github.io"
        "/auto-sklearn/stable/installation.html#windows-osx-compability" % sys.platform
    )

if sys.version_info < (3, 6):
    raise ValueError(
        "Unsupported python version %s found. Auto-sklearn requires Python "
        "3.6 or higher." % sys.version_info
    )
