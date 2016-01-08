# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import traceback

import numpy as np
from numpy.linalg import LinAlgError

from evaluation_util import get_dataset_getters, BaseEvaluatorTest

from autosklearn.evaluation import NestedCVEvaluator


N_TEST_RUNS = 10


class Dummy(object):
    pass


class NestedCVEvaluator_Test(BaseEvaluatorTest):
    _multiprocess_can_split_ = True


def generate(D, upper_error_bound):
    def run_test(self):
        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            D_ = copy.deepcopy(D)
            evaluator = NestedCVEvaluator(D_, None)

            evaluator.fit()

            err[i] = evaluator.predict()

            self.assertTrue(np.isfinite(err[i]))
            self.assertLessEqual(err[i], upper_error_bound)
            for model_idx in range(5):
                model = evaluator.outer_models[model_idx]
                self.assertIsNotNone(model)
                model = evaluator.inner_models[model_idx]
                self.assertIsNotNone(model)

    return run_test


for getter in get_dataset_getters():
    D, upper_error_bound = getter()
    setattr(NestedCVEvaluator_Test, 'test_%s' % str(getter),
            generate(D, upper_error_bound))