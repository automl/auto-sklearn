import queue

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, CLASSIFICATION_METRICS, \
    REGRESSION_METRICS


__all__ = [
    'get_last_result'
]


def get_last_result(queue_):
    stack = []
    while True:
        try:
            rval = queue_.get(timeout=1)
        except queue.Empty:
            break
        stack.append(rval)
    return stack.pop()
