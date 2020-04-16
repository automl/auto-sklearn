import queue


__all__ = [
    'read_queue'
]


def read_queue(queue_):
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
        return stack


def empty_queue(queue_):
    while True:
        try:
            queue_.get(block=False)
        except queue.Empty:
            break

    queue_.close()


def extract_learning_curve(stack, key=None):
    learning_curve = []
    for entry in stack:
        if key:
            learning_curve.append(entry['additional_run_info'][key])
        else:
            learning_curve.append(entry['loss'])
    return list(learning_curve)
