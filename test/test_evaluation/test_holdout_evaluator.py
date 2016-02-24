# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import shutil
import sys

import numpy as np

from autosklearn.constants import *
from autosklearn.evaluation.holdout_evaluator import HoldoutEvaluator
from autosklearn.util.pipeline import get_configuration_space

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters

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

    def test_file_output(self):
        self.output_dir = os.path.join(os.getcwd(), '.test')

        D = get_regression_datamanager()
        D.name = 'test'

        configuration_space = get_configuration_space(D.info)

        configuration = configuration_space.sample_configuration()
        evaluator = HoldoutEvaluator(D, self.output_dir, configuration,
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
                output_directory = os.path.join(os.getcwd(), '.%s' % testname)
                self.output_directory = output_directory

                err = np.zeros([N_TEST_RUNS])
                for i in range(N_TEST_RUNS):
                    D_ = copy.deepcopy(D)
                    evaluator = HoldoutEvaluator(D_, self.output_directory, None)

                    err[i] = evaluator.fit_predict_and_loss()[0]

                    self.assertTrue(np.isfinite(err[i]))
