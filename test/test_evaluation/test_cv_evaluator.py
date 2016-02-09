# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import sys
import numpy as np

from autosklearn.evaluation import CVEvaluator

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_dataset_getters, BaseEvaluatorTest

N_TEST_RUNS = 5


class CVEvaluator_Test(BaseEvaluatorTest):
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
                    evaluator = CVEvaluator(D_, output_directory, None)

                    evaluator.fit()
                    err[i] = evaluator.loss_and_predict()[0]

                    self.assertTrue(np.isfinite(err[i]))
                    self.assertLessEqual(err[i], upper_error_bound)
                    for model_idx in range(10):
                        model = evaluator.models[model_idx]
                        self.assertIsNotNone(model)

                    D_ = copy.deepcopy(D)
                    evaluator = CVEvaluator(D_, output_directory, None)
                    for j in range(5):
                        evaluator.partial_fit(j)
                        model = evaluator.models[j]
                        self.assertIsNotNone(model)
                    for j in range(5, 10):
                        model = evaluator.models[j]
                        self.assertIsNone(model)



# for getter in get_dataset_getters():
#     D, upper_error_bound = getter()
#     testname = '%s_%s' % (os.path.basename(__file__).
#                           replace('.pyc','').replace('.py', ''),
#                           getter.__name__)
#     output_directory = os.path.join(os.getcwd(), '._%s' % testname)
#     setattr(CVEvaluator_Test, 'test_%s' % testname,
#             generate(D, upper_error_bound, output_directory))
#     print(getattr(CVEvaluator_Test, 'test_%s' % testname))
