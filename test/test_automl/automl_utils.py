# -*- encoding: utf-8 -*-
import os

import numpy as np


def extract_msg_from_log(log_file):
    include_messages = ['INFO', 'DEBUG', 'WARN',
                        'CRITICAL', 'ERROR', 'FATAL']

    # There is a lot of content in the log files. Only
    # parsing the main message and ignore the metalearning
    # messages
    try:
        with open(log_file) as logfile:
            content = logfile.readlines()

        # Get the messages to debug easier!
        content = [x for x in content if any(
            msg in x for msg in include_messages
        ) and 'metalearning' not in x]

    except Exception as e:
        return str(e)
    return os.linesep.join(content)


def count_succeses(cv_results):
    return np.sum(
        [status in ['Success', 'Success (but do not advance to higher budget)']
         for status in cv_results['status']]
    )
