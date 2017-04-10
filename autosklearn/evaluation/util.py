import queue


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
    if len(stack) == 0:
        raise queue.Empty
    else:
        return stack.pop()
