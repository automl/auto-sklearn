# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import shutil

import numpy as np

from autosklearn.constants import *
from autosklearn.evaluation.holdout_evaluator import HoldoutEvaluator
from autosklearn.util.pipeline import get_configuration_space

from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters

N_TEST_RUNS = 10


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class HoldoutEvaluatorTest(BaseEvaluatorTest):
    _multiprocess_can_split_ = True

    def test_file_output(self):
        output_dir = os.path.join(os.getcwd(), '.test')

        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

        D, _ = get_regression_datamanager()
        D.name = 'test'

        configuration_space = get_configuration_space(D.info)

        while True:
            configuration = configuration_space.sample_configuration()
            evaluator = HoldoutEvaluator(D, configuration,
                                         with_predictions=True,
                                         all_scoring_functions=True,
                                         output_dir=output_dir,
                                         output_y_test=True)

            if not self._fit(evaluator):
                continue
            evaluator.predict()
            evaluator.file_output()

            self.assertTrue(os.path.exists(os.path.join(
                output_dir, '.auto-sklearn', 'true_targets_ensemble.npy')))
            break

    def test_predict_proba_binary_classification(self):
        D, _ = get_binary_classification_datamanager()

        class Dummy2(object):

            def predict_proba(self, y, batch_size=200):
                return np.array([[0.1, 0.9], [0.7, 0.3]])

        model = Dummy2()
        task_type = BINARY_CLASSIFICATION

        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['extra_trees'],
            include_preprocessors=['select_rates'])
        configuration = configuration_space.sample_configuration()

        evaluator = HoldoutEvaluator(D, configuration)
        pred = evaluator.predict_proba(None, model, task_type)
        expected = [[0.9], [0.3]]
        for i in range(len(expected)):
            self.assertEqual(expected[i], pred[i])


def generate(D, upper_error_bound):
    def run_test(self):

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, None)

            evaluator.fit()

            err[i] = evaluator.predict()

            self.assertTrue(np.isfinite(err[i]))
            self.assertLessEqual(err[i], upper_error_bound)

    return run_test


for getter in get_dataset_getters():
    D, upper_error_bound = getter()
    setattr(HoldoutEvaluatorTest, 'test_%s' % str(getter),
            generate(D, upper_error_bound))