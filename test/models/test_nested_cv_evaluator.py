# -*- encoding: utf-8 -*-
from __future__ import print_function

import copy
import os
import unittest

import numpy as np
from numpy.linalg import LinAlgError

from autosklearn.constants import *
from autosklearn.data_managers import CompetitionDataManager
from autosklearn.models.nested_cv_evaluator import NestedCVEvaluator
from autosklearn.models.paramsklearn import get_configuration_space
from ParamSklearn.util import get_dataset

N_TEST_RUNS = 10


class Dummy(object):
    pass


class NestedCVEvaluator_Test(unittest.TestCase):

    def test_evaluate_multiclass_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'acc_metric',
            'task': MULTICLASS_CLASSIFICATION,
            'is_sparse': False,
            'target_num': 3
        }
        D.data = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_valid': X_valid,
            'X_test': X_test
        }
        D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']

        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['ridge'],
            include_preprocessors=['select_rates'])

        err = np.zeros([N_TEST_RUNS])
        num_models_better_than_random = 0
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = NestedCVEvaluator(D_, configuration,
                                          with_predictions=True,
                                          all_scoring_functions=True)

            if not self._fit(evaluator):
                continue
            e_, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
                evaluator.predict()
            err[i] = e_['acc_metric']
            print(err[i], configuration['classifier'])
            print(e_['outer:bac_metric'], e_['bac_metric'])

            # Test the outer CV
            num_targets = len(np.unique(Y_train))
            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)
            # Test that ten models were trained
            self.assertEqual(len(evaluator.outer_models), 5)
            self.assertTrue(all([model is not None
                                 for model in evaluator.outer_models]))

            self.assertEqual(Y_optimization_pred.shape[0], Y_train.shape[0])
            self.assertEqual(Y_optimization_pred.shape[1], num_targets)
            self.assertEqual(Y_valid_pred.shape[0], Y_valid.shape[0])
            self.assertEqual(Y_valid_pred.shape[1], num_targets)
            self.assertEqual(Y_test_pred.shape[0], Y_test.shape[0])
            self.assertEqual(Y_test_pred.shape[1], num_targets)
            # Test some basic statistics of the predictions
            if err[i] < 0.5:
                self.assertTrue(0.3 < Y_valid_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_valid_pred.std(), 0.1)
                self.assertTrue(0.3 < Y_test_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_test_pred.std(), 0.1)
                num_models_better_than_random += 1

            # Test the inner CV
            self.assertEqual(len(evaluator.inner_models), 5)
            for fold in range(5):
                self.assertEqual(len(evaluator.inner_models[fold]), 5)
                self.assertTrue(all([model is not None
                                     for model in evaluator.inner_models[fold]
                                     ]))
                self.assertGreaterEqual(len(evaluator.outer_indices[fold][0]),
                                        75)
                for inner_fold in range(5):
                    self.assertGreaterEqual(
                        len(evaluator.inner_indices[fold][inner_fold][0]), 60)

        self.assertGreater(num_models_better_than_random, 9)

    def test_with_abalone(self):
        dataset = 'abalone'
        dataset_dir = os.path.join(os.path.dirname(__file__), '.datasets')
        D = CompetitionDataManager(dataset, dataset_dir)
        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['extra_trees'],
            include_preprocessors=['no_preprocessing'])

        errors = []
        for i in range(N_TEST_RUNS):
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = NestedCVEvaluator(D_, configuration,
                                          inner_cv_folds=2,
                                          outer_cv_folds=2)
            if not self._fit(evaluator):
                continue
            err = evaluator.predict()
            self.assertLess(err, 0.99)
            self.assertTrue(np.isfinite(err))
            errors.append(err)
        # This is a reasonable bound
        self.assertEqual(10, len(errors))
        self.assertLess(min(errors), 0.77)

    def _fit(self, evaluator):
        return self.__fit(evaluator.fit)

    def __fit(self, function_handle):
        """Allow us to catch known and valid exceptions for all evaluate
        scripts."""
        try:
            function_handle()
            return True
        except ValueError as e:
            if 'Floating-point under-/overflow occurred at epoch' in e.message or \
                    'removed all features' in e.message or \
                    'failed to create intent' in e.message:
                pass
            else:
                raise e
        except LinAlgError as e:
            if 'not positive definite, even with jitter' in e.message:
                pass
            else:
                raise e
        except AttributeError as e:
            # Some error in QDA
            if 'log' == e.message:
                pass
            else:
                raise e
        except RuntimeWarning as e:
            if 'invalid value encountered in sqrt' in e.message:
                pass
            elif 'divide by zero encountered in divide' in e.message:
                pass
            else:
                raise e
        except UserWarning as e:
            if 'FastICA did not converge' in e.message:
                pass
            else:
                raise e
