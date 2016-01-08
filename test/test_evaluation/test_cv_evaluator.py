# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy

import numpy as np

from autosklearn.evaluation import CVEvaluator

from evaluation_util import get_dataset_getters, BaseEvaluatorTest

N_TEST_RUNS = 5


class CVEvaluator_Test(BaseEvaluatorTest):
    _multiprocess_can_split_ = True


def generate(D, upper_error_bound):
    def run_test(self):
        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, None)

            evaluator.fit()

            err[i] = evaluator.predict()

            self.assertTrue(np.isfinite(err[i]))
            self.assertLessEqual(err[i], upper_error_bound)
            for model_idx in range(10):
                model = evaluator.models[model_idx]
                self.assertIsNotNone(model)

            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, None)
            for j in range(5):
                evaluator.partial_fit(j)
                model = evaluator.models[j]
                self.assertIsNotNone(model)
            for j in range(5, 10):
                model = evaluator.models[j]
                self.assertIsNone(model)

    return run_test


for getter in get_dataset_getters():
    D, upper_error_bound = getter()
    setattr(CVEvaluator_Test, 'test_%s' % str(getter),
            generate(D, upper_error_bound))