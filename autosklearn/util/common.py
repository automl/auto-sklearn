# -*- encoding: utf-8 -*-

import os
from sklearn.externals import six
import warnings

__all__ = [
    'check_pid',
    'warn_if_not_float'
]


def warn_if_not_float(X, estimator='This algorithm'):
    """Warning utility function to check that data type is floating point.
    Returns True if a warning was raised (i.e. the input is not float) and
    False otherwise, for easier input validation.
    """
    if not isinstance(estimator, six.string_types):
        estimator = estimator.__class__.__name__
    if X.dtype.kind != 'f':
        warnings.warn("%s assumes floating point values as input, "
                      "got %s" % (estimator, X.dtype))
        return True
    return False


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
