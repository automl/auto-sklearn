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

        # Check if there is a special placeholder value which tells us that
        # we don't have to wait until the queue times out in order to
        # retrieve the final value!
        if 'final_queue_element' in rval:
            del rval['final_queue_element']
            do_break = True
        else:
            do_break = False
        stack.append(rval)
        if do_break:
            break

    if len(stack) == 0:
        raise queue.Empty
    else:
        return stack.pop()
