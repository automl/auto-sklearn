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
                D = getter()
                output_directory = os.path.join(os.getcwd(), '.%s' % testname)
                err = np.zeros([N_TEST_RUNS])
                for i in range(N_TEST_RUNS):
                    D_ = copy.deepcopy(D)
                    evaluator = CVEvaluator(D_, output_directory, None)

                    err[i] = evaluator.fit_predict_and_loss()[0]

                    self.assertTrue(np.isfinite(err[i]))
                    self.assertEqual(err[i], 1.0)
                    for model_idx in range(10):
                        indices = evaluator.indices[model_idx]
                        self.assertIsNotNone(indices)

                    D_ = copy.deepcopy(D)
                    evaluator = CVEvaluator(D_, output_directory, None)
                    for j in range(5):
                        evaluator.partial_fit_predict_and_loss(j)
                        indices = evaluator.indices[j]
                        self.assertIsNotNone(indices)
                    for j in range(5, 10):
                        indices = evaluator.indices[j]
                        self.assertIsNone(indices)
