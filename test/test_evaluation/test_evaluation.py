import os
import logging
import shutil
import sys
import time
import unittest
import unittest.mock

import numpy as np

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)

import pynisher
from smac.tae.execute_ta_run import StatusType

from evaluation_util import get_multiclass_classification_datamanager
from autosklearn.evaluation import ExecuteTaFuncWithQueue


def safe_eval_success_mock(*args, **kwargs):
    queue = kwargs['queue']
    queue.put((StatusType.SUCCESS, 0.5, 0.12345, ''))


class BackendMock(object):
    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.datamanager = get_multiclass_classification_datamanager()
        self.tmp = os.path.join(os.getcwd(), '.test_evaluation')
        self.logger = logging.getLogger()

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
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

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

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout(self, pynisher_mock):
        pynisher_mock.side_effect = safe_eval_success_mock
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger)
        info = ta.run(None, cutoff=30, memory_limit=3000)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[1], 0.5)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = None
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger)
        info = ta.run(None, cutoff=30, memory_limit=3000)
        self.assertEqual(info[0], StatusType.CRASHED)
        self.assertEqual(info[1], 2.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.side_effect = MemoryError
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger)
        info = ta.run(None, cutoff=30, memory_limit=3000)
        self.assertEqual(info[0], StatusType.MEMOUT)
        self.assertEqual(info[1], 2.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        pynisher_mock.side_effect = pynisher.TimeoutException
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger)
        info = ta.run(None, cutoff=30, memory_limit=3000)
        self.assertEqual(info[0], StatusType.TIMEOUT)
        self.assertEqual(info[1], 2.0)
        self.assertIsInstance(info[2], float)
