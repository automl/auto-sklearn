import os
import shutil
import sys
import unittest

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)

from smac.tae.execute_ta_run import StatusType

from evaluation_util import get_multiclass_classification_datamanager
from autosklearn.evaluation import eval_with_limits


def safe_eval_success_mock(*args):
    def decorator(*args, **kwargs):
        queue = kwargs['queue']
        queue.put((0.1, 1.0, 1, '', StatusType.SUCCESS))
    return decorator


def safe_eval_fail_silently_mock(*args):
    def decorator(*args, **kwargs):
        pass
    return decorator


def safe_eval_memory_error_mock(*args):
    def decorator(*args, **kwargs):
        raise MemoryError()
    return decorator


def safe_eval_timeout_mock(*args):
    def decorator(*args, **kwargs):
        raise ValueError()
    return decorator


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

    @mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout(self, pynisher_mock):
        pynisher_mock.return_value = safe_eval_success_mock
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info, (0.1, 1.0, 1, '', StatusType.SUCCESS))

    @mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = safe_eval_fail_silently_mock
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.CRASHED)

    @mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.return_value = safe_eval_memory_error_mock
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.MEMOUT)

    @mock.patch('pynisher.enforce_limits')
    @mock.patch('time.time')
    def test_eval_with_limits_holdout_fail_timeout(self, time_mock,
                                                   pynisher_mock):
        time_mock.side_effect = [i * 70 for i in range(1000)]
        pynisher_mock.return_value = safe_eval_timeout_mock
        info = eval_with_limits(self.datamanager, self.tmp, None, 1, 1,
                                'holdout', {}, 3000, 30)
        self.assertEqual(info[1], 2.0)
        self.assertEqual(info[4], StatusType.TIMEOUT)
