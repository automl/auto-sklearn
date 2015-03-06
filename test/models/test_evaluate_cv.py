import copy
import unittest
import numpy as np
import os

from autosklearn.models.evaluate_cv import CVEvaluator
from autosklearn.models.evaluate import calculate_score
from autosklearn.models.paramsklearn import get_configuration_space
from autosklearn.data.split_data import split_data
from ParamSklearn.util import get_dataset
from HPOlibConfigSpace.random_sampler import RandomSampler

N_TEST_RUNS = 50


class Dummy(object):
    pass


class Test(unittest.TestCase):
    def test_evaluate_multiclass_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {'metric': 'bac_metric', 'task': 'multiclass.classification',
                  'is_sparse': False}
        D.data = {'X_train': X_train, 'Y_train': Y_train,
                  'X_valid': X_valid, 'X_test': X_test}
        D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']

        configuration_space = get_configuration_space(D.info)
        sampler = RandomSampler(configuration_space, 1)

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print "Evaluate configuration: %d; result:" % i,
            configuration = sampler.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, configuration,
                                    splitting_function=split_data,
                                    with_predictions=True)

            if not self._fit(evaluator):
                print
                continue
            e_, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
                evaluator.predict()
            err[i] = e_
            print err[i]

            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)
            self.assertEqual(len(evaluator.models), 10)
            self.assertEqual(Y_optimization_pred.shape[0], Y_train.shape[0])
            self.assertEqual(Y_valid_pred.shape[0], Y_valid.shape[0])
            self.assertEqual(Y_test_pred.shape[0], Y_test.shape[0])

    def _fit(self, evaluator):
        """Allow us to catch known and valid exceptions for all evaluate
        scripts."""
        try:
            evaluator.fit()
            return True
        except ValueError as e:
            if "Floating-point under-/overflow occurred at epoch" in e.message:
                return False
            else:
                print evaluator.configuration
                raise e