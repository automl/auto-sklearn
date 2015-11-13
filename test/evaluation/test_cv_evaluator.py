# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import functools
import os
import unittest

import numpy as np
from numpy.linalg import LinAlgError

from autosklearn.constants import *
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation.cv_evaluator import CVEvaluator
from autosklearn.util.paramsklearn import get_configuration_space
from ParamSklearn.util import get_dataset

N_TEST_RUNS = 10


class Dummy(object):
    pass


class CVEvaluator_Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_evaluate_multiclass_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': BAC_METRIC,
            'task': MULTICLASS_CLASSIFICATION,
            'is_sparse': False,
            'label_num': 3
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
            include_estimators=['extra_trees'],
            include_preprocessors=['select_rates'])

        err = np.zeros([N_TEST_RUNS])
        num_models_better_than_random = 0
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, configuration, with_predictions=True)

            if not self._fit(evaluator):
                print()
                continue
            e_, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
                evaluator.predict()
            err[i] = e_
            print(err[i], configuration['classifier:__choice__'])

            num_targets = len(np.unique(Y_train))
            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)
            # Test that ten models were trained
            self.assertEqual(len(evaluator.models), 10)
            self.assertEqual(Y_optimization_pred.shape[0], Y_train.shape[0])
            self.assertEqual(Y_optimization_pred.shape[1], num_targets)
            self.assertEqual(Y_valid_pred.shape[0], Y_valid.shape[0])
            self.assertEqual(Y_valid_pred.shape[1], num_targets)
            self.assertEqual(Y_test_pred.shape[0], Y_test.shape[0])
            self.assertEqual(Y_test_pred.shape[1], num_targets)
            # Test some basic statistics of the dataset
            if err[i] < 0.5:
                self.assertTrue(0.3 < Y_valid_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_valid_pred.std(), 0.01)
                self.assertTrue(0.3 < Y_test_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_test_pred.std(), 0.01)
                num_models_better_than_random += 1
        self.assertGreater(num_models_better_than_random, 5)

    def test_evaluate_multiclass_classification_partial_fit(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': BAC_METRIC,
            'task': MULTICLASS_CLASSIFICATION,
            'is_sparse': False,
            'label_num': 3
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
            include_estimators=['lda'],
            include_preprocessors=['select_rates'])

        err = np.zeros([N_TEST_RUNS])
        num_models_better_than_random = 0
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, configuration, with_predictions=True)

            if not self._partial_fit(evaluator, fold=i % 10):
                print()
                continue
            e_, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
                evaluator.predict()
            err[i] = e_
            print(err[i], configuration['classifier:__choice__'])

            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)
            # Test that only one model was trained
            self.assertEqual(len(evaluator.models), 10)
            self.assertEqual(1, np.sum([True if model is not None else False
                                        for model in evaluator.models]))
            self.assertLess(Y_optimization_pred.shape[0], 13)
            self.assertEqual(Y_valid_pred.shape[0], Y_valid.shape[0])
            self.assertEqual(Y_test_pred.shape[0], Y_test.shape[0])
            # Test some basic statistics of the dataset
            if err[i] < 0.5:
                self.assertTrue(0.3 < Y_valid_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_valid_pred.std(), 0.01)
                self.assertTrue(0.3 < Y_test_pred.mean() < 0.36666)
                self.assertGreaterEqual(Y_test_pred.std(), 0.01)
                num_models_better_than_random += 1
        self.assertGreaterEqual(num_models_better_than_random, 5)

    def test_with_abalone(self):
        dataset = 'abalone'
        dataset_path = os.path.join(os.path.dirname(__file__), '.datasets',
                                    dataset)
        D = CompetitionDataManager(dataset_path)
        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['extra_trees'],
            include_preprocessors=['no_preprocessing'])

        errors = []
        for i in range(N_TEST_RUNS):
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = CVEvaluator(D_, configuration, cv_folds=3)
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

    def _partial_fit(self, evaluator, fold):
        partial_fit = functools.partial(evaluator.partial_fit, fold=fold)
        return self.__fit(partial_fit)

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
