# -*- encoding: utf-8 -*-
import os
import sys
import warnings

from autosklearn.__version__ import __version__  # noqa (imported but unused)

# Check operating system compatibility
if os.name != "posix":
    warnings.warn(
        "Detected operating system: %s. auto-sklearn officially supports only "
        "POSIX systems (Linux/macOS). Some features may not work correctly on Windows. "
        "Please check the compatibility information: https://automl.github.io"
        "/auto-sklearn/stable/installation.html#windows-osx-compatibility" % sys.platform,
        UserWarning,
        stacklevel=2,
    )

# Check Python version compatibility
if sys.version_info < (3, 7):
    raise ValueError(
        "Unsupported python version %s found. Auto-sklearn requires Python "
        "3.7 or higher." % sys.version_info
    )
