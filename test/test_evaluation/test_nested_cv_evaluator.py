# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import sys

import numpy as np

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_dataset_getters, BaseEvaluatorTest

from autosklearn.evaluation import NestedCVEvaluator


N_TEST_RUNS = 10


class Dummy(object):
    pass


class NestedCVEvaluator_Test(BaseEvaluatorTest):
    _multiprocess_can_split_ = True

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)
            with self.subTest(testname):
                D, upper_error_bound = getter()
                output_directory = os.path.join(os.getcwd(), '.%s' % testname)
                err = np.zeros([N_TEST_RUNS])
                for i in range(N_TEST_RUNS):
                    D_ = copy.deepcopy(D)
                    evaluator = NestedCVEvaluator(D_, output_directory, None)

                    evaluator.fit()

                    err[i] = evaluator.loss_and_predict()[0]

                    self.assertTrue(np.isfinite(err[i]))
                    self.assertLessEqual(err[i], upper_error_bound)
                    for model_idx in range(5):
                        model = evaluator.outer_models[model_idx]
                        self.assertIsNotNone(model)
                        model = evaluator.inner_models[model_idx]
                        self.assertIsNotNone(model)

# def generate(D, upper_error_bound, output_directory):
#     def run_test(self):
#         self.output_directory = output_directory
#
#         err = np.zeros([N_TEST_RUNS])
#         for i in range(N_TEST_RUNS):
#             D_ = copy.deepcopy(D)
#             evaluator = NestedCVEvaluator(D_, self.output_directory, None)
#
#             evaluator.fit()
#
#             err[i] = evaluator.predict()
#
#             self.assertTrue(np.isfinite(err[i]))
#             self.assertLessEqual(err[i], upper_error_bound)
#             for model_idx in range(5):
#                 model = evaluator.outer_models[model_idx]
#                 self.assertIsNotNone(model)
#                 model = evaluator.inner_models[model_idx]
#                 self.assertIsNotNone(model)
#
#     return run_test
#
#
# for getter in get_dataset_getters():
#     D, upper_error_bound = getter()
#     testname = '%s_%s' % (os.path.basename(__file__).
#                           replace('.pyc', '').replace('.py', ''),
#                           getter.__name__)
#     output_directory = os.path.join(os.getcwd(), '._%s' % testname)
#     setattr(NestedCVEvaluator_Test, 'test_%s' % testname,
#             generate(D, upper_error_bound, output_directory))