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
from smac.stats.stats import Stats

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
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = 10
        scenario_mock.algo_runs_timelimit = 1000
        scenario_mock.ta_run_limit = 100
        self.scenario = scenario_mock
        stats = Stats(scenario_mock)
        stats.start_timing()
        self.stats = stats

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
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        info = ta.run(None, cutoff=30)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[1], 0.5)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        ta.run(None, cutoff=30)
        self.assertEqual(pynisher_mock.call_args[1]['wall_time_in_s'], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]['wall_time_in_s'], int)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_zero_or_negative_cutoff(self, pynisher_mock):
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats)
        self.scenario.wallclock_limit = 5
        info = ta.start(None, instance=None, cutoff=10)
        fixture = (StatusType.ABORT, np.nan, 0, {"misc": "exhausted bugdet -- ABORT"})
        self.assertEqual(info, fixture)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats)
        self.stats.ta_runs = 1
        ta.start(None, cutoff=30, instance=None)
        self.assertEqual(pynisher_mock.call_args[1]['wall_time_in_s'], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]['wall_time_in_s'], int)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = None
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        info = ta.run(None, cutoff=30)
        self.assertEqual(info[0], StatusType.CRASHED)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.side_effect = MemoryError
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        info = ta.run(None, cutoff=30)
        self.assertEqual(info[0], StatusType.MEMOUT)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        pynisher_mock.side_effect = pynisher.TimeoutException
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        info = ta.run(None, cutoff=30)
        self.assertEqual(info[0], StatusType.TIMEOUT)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_eval_with_limits_holdout_timeout_with_results_in_queue(self, pynisher_mock):
        def side_effect(**kwargs):
            queue = kwargs['queue']
            queue.put((StatusType.SUCCESS, 0.5, 0.12345, ''))
        pynisher_mock.side_effect = side_effect
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3000)
        info = ta.run(None, cutoff=30)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[1], 0.5)
        self.assertIsInstance(info[2], float)
