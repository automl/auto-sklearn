import os
import shutil
import sys
import time
import unittest

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)

import pynisher
from smac.tae.execute_ta_run import StatusType

from evaluation_util import get_multiclass_classification_datamanager
from autosklearn.evaluation import eval_with_limits


def safe_eval_success_mock(*args, **kwargs):
    queue = kwargs['queue']
    queue.put((0.1, 1.0, 1, '', StatusType.SUCCESS))


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.datamanager = get_multiclass_classification_datamanager()
        self.tmp = os.path.join(os.getcwd(), '.test_evaluation')

        try:
            shutil.rmtree(self.tmp)
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp)
        except:
            pass

    def test_pynisher_memory_error(self):
        def fill_memory():
            lst = []
            for i in range(1000000):
                lst.append(i)
            return lst

        safe_eval = pynisher.enforce_limits(mem_in_mb=1)(fill_memory)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.MemorylimitException)

    def test_pynisher_timeout(self):
        def run_over_time():
            time.sleep(2)

        safe_eval = pynisher.enforce_limits(wall_time_in_s=1,
                                            grace_period_in_s=0)(run_over_time)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.TimeoutException)

    @mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout(self, pynisher_mock):
        pynisher_mock.side_effect = safe_eval_success_mock
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 1.0)
        self.assertEqual(info[2], 1)
        self.assertEqual(info[4], StatusType.SUCCESS)

    @mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = None
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.CRASHED)

    @mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.side_effect = MemoryError
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.MEMOUT)

    @mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        pynisher_mock.side_effect = pynisher.TimeoutException
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.TIMEOUT)
