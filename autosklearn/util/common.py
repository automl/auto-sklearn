# -*- encoding: utf-8 -*-


import os
import sys


__all__ = [
    'check_pid'
]


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
