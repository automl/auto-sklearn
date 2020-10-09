# -*- encoding: utf-8 -*-
import os
import shutil
import time
import unittest

import numpy as np

from autosklearn.util.backend import create


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


class Base(unittest.TestCase):
    _multiprocess_can_split_ = True
    """All tests which are a subclass of this must define their own output
    directory and call self._setUp."""

    def setUp(self):
        self.test_dir = os.path.dirname(__file__)

        try:
            os.environ['TRAVIS']
            self.travis = True
        except Exception:
            self.travis = False

    def _setUp(self, dir):
        if os.path.exists(dir):
            for i in range(10):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    def _create_backend(self, test_name, delete_tmp_folder_after_terminate=True,
                        delete_output_folder_after_terminate=True):
        tmp = os.path.join(self.test_dir, '..', '.tmp._%s' % test_name)
        output = os.path.join(self.test_dir, '..', '.output._%s' % test_name)
        # Make sure the folders we wanna create do not already exist.
        self._setUp(tmp)
        self._setUp(output)
        backend = create(
            tmp,
            output,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            )
        return backend

    def _tearDown(self, dir):
        """
        Delete the temporary and the output directories manually
        in case they are not deleted.
        """
        if os.path.exists(dir):
            for i in range(10):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)
