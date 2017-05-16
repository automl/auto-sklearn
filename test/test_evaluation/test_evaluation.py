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
from smac.tae.execute_ta_run import StatusType, FirstRunCrashedException, \
    BudgetExhaustedException
from smac.stats.stats import Stats
import sklearn.model_selection

from evaluation_util import get_multiclass_classification_datamanager
from autosklearn.constants import *
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.metrics import accuracy


def safe_eval_success_mock(*args, **kwargs):
    queue = kwargs['queue']
    queue.put({'status': StatusType.SUCCESS,
               'loss': 0.5,
               'additional_run_info': ''})


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

    ############################################################################
    # pynisher tests
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

    ############################################################################
    # Test ExecuteTaFuncWithQueue.start()
    @unittest.mock.patch('autosklearn.evaluation.train_evaluator.eval_holdout')
    def test_eval_with_limits_holdout(self, pynisher_mock):
        pynisher_mock.side_effect = safe_eval_success_mock
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        info = ta.start(None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[1], 0.5)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_zero_or_negative_cutoff(self, pynisher_mock):
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    metric=accuracy)
        self.scenario.wallclock_limit = 5
        self.stats.ta_runs += 1
        self.assertRaises(BudgetExhaustedException, ta.start, None,
                          instance=None, cutoff=9)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    metric=accuracy)
        self.stats.ta_runs = 1
        ta.start(None, cutoff=30, instance=None)
        self.assertEqual(pynisher_mock.call_args[1]['wall_time_in_s'], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]['wall_time_in_s'], int)

    @unittest.mock.patch('autosklearn.evaluation.train_evaluator.eval_holdout')
    def test_eval_with_limits_holdout_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = None
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)

        self.assertRaisesRegex(FirstRunCrashedException,
                               "First run crashed, abort. \(To prevent this, "
                               "toggle the "
                               "'abort_on_first_run_crash'-option!\)",
                               ta.start, config=None, instance=None, cutoff=30)

        self.stats.ta_runs += 1
        info = ta.start(config=None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.CRASHED)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)
        self.assertEqual(info[3], {'error': "Result queue is empty"})

    @unittest.mock.patch('autosklearn.evaluation.train_evaluator.eval_holdout')
    def test_eval_with_limits_holdout_fail_memory_error(self, pynisher_mock):
        pynisher_mock.side_effect = MemoryError
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        info = ta.start(None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.MEMOUT)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        info = ta.start(config=None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.TIMEOUT)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_timeout_with_results_in_queue(self, pynisher_mock):
        def side_effect(**kwargs):
            queue = kwargs['queue']
            queue.put({'status': StatusType.SUCCESS,
                       'loss': 0.5,
                       'additional_run_info': {}})
        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.side_effect = side_effect
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30

        # Test for a succesful run
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        info = ta.start(None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[1], 0.5)
        self.assertIsInstance(info[2], float)

        # And a crashed run which is in the queue
        def side_effect(**kwargs):
            queue = kwargs['queue']
            queue.put({'status': StatusType.CRASHED,
                       'loss': 2.0,
                       'additional_run_info': {}})
        m2.side_effect = side_effect
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        info = ta.start(None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.CRASHED)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)

    @unittest.mock.patch('autosklearn.evaluation.train_evaluator.eval_holdout')
    def test_eval_with_limits_holdout_2(self, eval_houldout_mock):
        def side_effect(*args, **kwargs):
            queue = kwargs['queue']
            queue.put({'status': StatusType.SUCCESS,
                       'loss': 0.5,
                       'additional_run_info': kwargs['instance']})
        eval_houldout_mock.side_effect = side_effect
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        self.scenario.wallclock_limit = 180
        instance = "{'subsample': 30}"
        info = ta.start(None, cutoff=30, instance=instance)
        self.assertEqual(info[0], StatusType.SUCCESS)
        self.assertEqual(info[-1], instance)

    @unittest.mock.patch('autosklearn.evaluation.train_evaluator.eval_holdout')
    def test_exception_in_target_function(self, eval_holdout_mock):
        eval_holdout_mock.side_effect = ValueError
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), autosklearn_seed=1,
                                    resampling_strategy='holdout',
                                    logger=self.logger,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy)
        self.stats.ta_runs += 1
        info = ta.start(None, instance=None, cutoff=30)
        self.assertEqual(info[0], StatusType.CRASHED)
        self.assertEqual(info[1], 1.0)
        self.assertIsInstance(info[2], float)
        self.assertEqual(info[3]['error'], 'ValueError()')
        self.assertIn('traceback', info[3])

    ############################################################################
    # Test obtaining a splitter object from scikit-learn
    def test_get_splitter(self):
        ta_args = dict(backend=BackendMock(), autosklearn_seed=1,
                       logger=self.logger, stats=self.stats, memory_limit=3072,
                       metric=accuracy)
        D = unittest.mock.Mock(spec=AbstractDataManager)
        D.data = dict(Y_train=np.array([0, 0, 0, 1, 1, 1]))
        D.info = dict(task=BINARY_CLASSIFICATION)

        # holdout, binary classification
        ta = ExecuteTaFuncWithQueue(resampling_strategy='holdout', **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection.StratifiedShuffleSplit)

        # holdout, binary classification, fallback to shuffle split
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 2])
        ta = ExecuteTaFuncWithQueue(resampling_strategy='holdout', **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.ShuffleSplit)

        # cv, binary classification
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        ta = ExecuteTaFuncWithQueue(resampling_strategy='cv', folds=5,
                                    **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.StratifiedKFold)

        # cv, binary classification, no fallback anticipated
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 2])
        ta = ExecuteTaFuncWithQueue(resampling_strategy='cv', folds=5,
                                    **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.StratifiedKFold)

        # regression, shuffle split
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        ta = ExecuteTaFuncWithQueue(resampling_strategy='holdout', **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.ShuffleSplit)

        # regression cv, KFold
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        ta = ExecuteTaFuncWithQueue(resampling_strategy='cv', folds=5,
                                    **ta_args)
        cv = ta.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)
