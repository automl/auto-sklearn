# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import multiprocessing
import os
import shutil
import sys
import unittest

import numpy as np

from autosklearn.constants import *
from autosklearn.evaluation.holdout_evaluator import HoldoutEvaluator, \
    eval_holdout, eval_iterative_holdout
from autosklearn.util import backend
from autosklearn.util.pipeline import get_configuration_space

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters, \
    get_multiclass_classification_datamanager

N_TEST_RUNS = 10


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class HoldoutEvaluatorTest(BaseEvaluatorTest):
    _multiprocess_can_split_ = True

    def teardown(self):
        try:
            shutil.rmtree(self.output_dir)
        except Exception:
            pass

        for output_dir in self.output_directories:
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass

    def test_file_output(self):
        self.output_dir = os.path.join(os.getcwd(), '.test_file_output')

        D = get_regression_datamanager()
        D.name = 'test'

        configuration_space = get_configuration_space(D.info)

        configuration = configuration_space.sample_configuration()
        backend_api = backend.create(self.output_dir, self.output_dir)
        evaluator = HoldoutEvaluator(D, backend_api, configuration,
                                     with_predictions=True,
                                     all_scoring_functions=True,
                                     output_y_test=True)

        loss, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
            evaluator.fit_predict_and_loss()
        evaluator.file_output(loss, Y_optimization_pred, Y_valid_pred,
                              Y_test_pred)

        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, '.auto-sklearn', 'true_targets_ensemble.npy')))

    def test_predict_proba_binary_classification(self):
        self.output_dir = os.path.join(os.getcwd(),
                                       '.test_predict_proba_binary_classification')
        D = get_binary_classification_datamanager()

        class Dummy2(object):

            def predict_proba(self, y, batch_size=200):
                return np.array([[0.1, 0.9]] * 23)

            def fit(self, X, y):
                return self

        model = Dummy2()

        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['extra_trees'],
            include_preprocessors=['select_rates'])
        configuration = configuration_space.sample_configuration()

        evaluator = HoldoutEvaluator(D, self.output_dir, configuration)
        evaluator.model = model
        loss, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
            evaluator.fit_predict_and_loss()

        for i in range(23):
            self.assertEqual(0.9, Y_optimization_pred[i][1])

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)
            with self.subTest(testname):
                D = getter()
                output_directory = os.path.join(os.path.dirname(__file__),
                                                '.%s' % testname)
                self.output_directory = output_directory
                self.output_directories.append(output_directory)

                err = np.zeros([N_TEST_RUNS])
                for i in range(N_TEST_RUNS):
                    D_ = copy.deepcopy(D)
                    evaluator = HoldoutEvaluator(D_, self.output_directory, None)

                    err[i] = evaluator.fit_predict_and_loss()[0]

                    self.assertTrue(np.isfinite(err[i]))



class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.queue = multiprocessing.Queue()
        self.configuration = get_configuration_space(
            {'task': MULTICLASS_CLASSIFICATION,
             'is_sparse': False}).get_default_configuration()
        self.data = get_multiclass_classification_datamanager()
        self.tmp_dir = os.path.join(os.path.dirname(__file__),
                                    '.test_holdout_functions')

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def test_eval_holdout(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_holdout(self.queue, self.configuration, self.data, backend_api,
                     1, 1, None, True, False, True)
        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.05)
        self.assertEqual(info[2], 1)
        self.assertNotIn('bac_metric', info[3])

    def test_eval_holdout_all_loss_functions(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_holdout(self.queue, self.configuration, self.data, backend_api,
                     1, 1, None, True, True, True)
        info = self.queue.get()
        self.assertIn('f1_metric: 0.0480549199085;pac_metric: 0.135572680594;'
                      'acc_metric: 0.0454545454545;auc_metric: 0.0;'
                      'bac_metric: 0.05;duration: ', info[3])
        self.assertAlmostEqual(info[1], 0.05)
        self.assertEqual(info[2], 1)

    def test_eval_holdout_on_subset(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_holdout(self.queue, self.configuration, self.data,
                     backend_api, 1, 1, 43, True, False, True)
        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.1)
        self.assertEqual(info[2], 1)

    def test_eval_holdout_iterative_fit_no_timeout(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_iterative_holdout(self.queue, self.configuration, self.data,
                               backend_api, 1, 1, None, True, False, True)
        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.05)
        self.assertEqual(info[2], 1)

    def test_eval_holdout_iterative_fit_on_subset_no_timeout(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_iterative_holdout(self.queue, self.configuration,
                               self.data, backend_api, 1, 1, 43, True, False,
                               True)

        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.1)
        self.assertEqual(info[2], 1)
