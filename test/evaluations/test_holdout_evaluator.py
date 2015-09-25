# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import shutil
import sys
import traceback
import unittest

import numpy as np
from numpy.linalg import LinAlgError
import sklearn.datasets

from ParamSklearn.util import get_dataset

from autosklearn.constants import *
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation.holdout_evaluator import HoldoutEvaluator
from autosklearn.util.data import convert_to_bin
from autosklearn.util.paramsklearn import get_configuration_space

N_TEST_RUNS = 10


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class HoldoutEvaluator_Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_evaluate_multiclass_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')
        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'bac_metric',
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
            include_estimators=['ridge'],
            include_preprocessors=['select_rates'])

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, configuration)

            if not self._fit(evaluator):
                continue
            err[i] = evaluator.predict()
            print(err[i])

            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)

        print('Number of times it was worse than random guessing:' +
              str(np.sum(err > 1)))

    def test_evaluate_multiclass_classification_all_metrics(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')
        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'bac_metric',
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
            include_estimators=['ridge'],
            include_preprocessors=['select_rates'])

        # Test all scoring functions
        err = []
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, configuration,
                                         all_scoring_functions=True)
            if not self._fit(evaluator):
                continue

            err.append(evaluator.predict())
            print(err[-1])

            self.assertIsInstance(err[-1], dict)
            for key in err[-1]:
                self.assertEqual(len(err[-1]), 5)
                self.assertTrue(np.isfinite(err[-1][key]))
                self.assertGreaterEqual(err[-1][key], 0.0)

        print('Number of times it was worse than random guessing:' +
              str(np.sum(err > 1)))

    def test_evaluate_multilabel_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')
        Y_train = np.array(convert_to_bin(Y_train, 3))
        Y_train[:, -1] = 1
        Y_test = np.array(convert_to_bin(Y_test, 3))
        Y_test[:, -1] = 1

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'f1_metric',
            'task': MULTILABEL_CLASSIFICATION,
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
            include_estimators=['random_forest'],
            include_preprocessors=['no_preprocessing'])

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, configuration)
            if not self._fit(evaluator):
                continue
            err[i] = evaluator.predict()
            print(err[i])

            self.assertTrue(np.isfinite(err[i]))
            self.assertGreaterEqual(err[i], 0.0)

        print('Number of times it was worse than random guessing:' +
              str(np.sum(err > 1)))

    def test_evaluate_binary_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        eliminate_class_two = Y_train != 2
        X_train = X_train[eliminate_class_two]
        Y_train = Y_train[eliminate_class_two]

        eliminate_class_two = Y_test != 2
        X_test = X_test[eliminate_class_two]
        Y_test = Y_test[eliminate_class_two]

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'auc_metric',
            'task': BINARY_CLASSIFICATION,
            'is_sparse': False,
            'label_num': 2
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
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, configuration)

            if not self._fit(evaluator):
                continue
            err[i] = evaluator.predict()
            self.assertTrue(np.isfinite(err[i]))
            print(err[i])

            self.assertGreaterEqual(err[i], 0.0)

        print('Number of times it was worse than random guessing:' +
              str(np.sum(err > 1)))

    def test_evaluate_regression(self):
        X_train, Y_train, X_test, Y_test = get_dataset('boston')

        X_valid = X_test[:200, ]
        Y_valid = Y_test[:200, ]
        X_test = X_test[200:, ]
        Y_test = Y_test[200:, ]

        D = Dummy()
        D.info = {
            'metric': 'r2_metric',
            'task': REGRESSION,
            'is_sparse': False,
            'label_num': 1
        }
        D.data = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_valid': X_valid,
            'X_test': X_test
        }
        D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical',
                       'numerical', 'numerical', 'numerical', 'numerical',
                       'numerical', 'numerical', 'numerical']

        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['random_forest'],
            include_preprocessors=['no_preprocessing'])

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print('Evaluate configuration: %d; result:' % i)
            configuration = configuration_space.sample_configuration()
            D_ = copy.deepcopy(D)
            evaluator = HoldoutEvaluator(D_, configuration)
            if not self._fit(evaluator):
                continue
            err[i] = evaluator.predict()
            self.assertTrue(np.isfinite(err[i]))
            print(err[i])

            self.assertGreaterEqual(err[i], 0.0)

        print('Number of times it was worse than random guessing:' +
              str(np.sum(err > 1)))

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
            evaluator = HoldoutEvaluator(D_, configuration)
            if not self._fit(evaluator):
                continue
            err = evaluator.predict()
            self.assertLess(err, 0.99)
            self.assertTrue(np.isfinite(err))
            errors.append(err)
        # This is a reasonable bound
        self.assertEqual(10, len(errors))
        self.assertLess(min(errors), 0.77)

    def test_5000_classes(self):
        weights = ([0.0002] * 4750) + ([0.0001] * 250)
        X, Y = sklearn.datasets.make_classification(n_samples=10000,
                                                    n_features=20,
                                                    n_classes=5000,
                                                    n_clusters_per_class=1,
                                                    n_informative=15,
                                                    n_redundant=5,
                                                    n_repeated=0,
                                                    weights=weights,
                                                    flip_y=0,
                                                    class_sep=1.0,
                                                    hypercube=True,
                                                    shift=None,
                                                    scale=1.0,
                                                    shuffle=True,
                                                    random_state=1)

        self.assertEqual(250, np.sum(np.bincount(Y) == 1))
        D = Dummy()
        D.info = {
            'metric': 'acc_metric',
            'task': MULTICLASS_CLASSIFICATION,
            'is_sparse': False,
            'label_num': 1
        }
        D.data = {'X_train': X, 'Y_train': Y, 'X_valid': X, 'X_test': X}
        D.feat_type = ['numerical'] * 5000

        configuration_space = get_configuration_space(
            D.info,
            include_estimators=['extra_trees'],
            include_preprocessors=['no_preprocessing'])
        configuration = configuration_space.sample_configuration()
        D_ = copy.deepcopy(D)
        evaluator = HoldoutEvaluator(D_, configuration)
        evaluator.fit()

    def _fit(self, evaluator):
        """Allow us to catch known and valid exceptions for all evaluate
        scripts."""
        try:
            evaluator.fit()
            return True
        except ValueError as e:
            if 'Floating-point under-/overflow occurred at epoch' in e.message or \
                    'removed all features' in e.message or \
                    'failed to create intent' in e.message:
                pass
            else:
                traceback.print_exc()
                raise e
        except LinAlgError as e:
            if 'not positive definite, even with jitter' in e.message:
                pass
            else:
                traceback.print_exc()
                raise e
        except AttributeError as e:
            # Some error in QDA
            if 'log' == e.message:
                pass
            else:
                traceback.print_exc()
                raise e
        except RuntimeWarning as e:
            if 'invalid value encountered in sqrt' in e.message:
                pass
            elif 'divide by zero encountered in divide' in e.message:
                pass
            else:
                traceback.print_exc()
                raise e
        except UserWarning as e:
            if 'FastICA did not converge' in e.message:
                pass
            else:
                traceback.print_exc()
                raise e

    def test_file_output(self):
        output_dir = os.path.join(os.getcwd(), '.test')

        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

        X_train, Y_train, X_test, Y_test = get_dataset('boston')
        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        D = Dummy()
        D.info = {
            'metric': 'r2_metric',
            'task': REGRESSION,
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
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        eliminate_class_two = Y_train != 2
        X_train = X_train[eliminate_class_two]
        Y_train = Y_train[eliminate_class_two]

        eliminate_class_two = Y_test != 2
        X_test = X_test[eliminate_class_two]
        Y_test = Y_test[eliminate_class_two]

        X_valid = X_test[:25, ]
        Y_valid = Y_test[:25, ]
        X_test = X_test[25:, ]
        Y_test = Y_test[25:, ]

        class Dummy2(object):

            def predict_proba(self, y, batch_size=200):
                return np.array([[0.1, 0.9], [0.7, 0.3]])

        model = Dummy2()
        task_type = BINARY_CLASSIFICATION

        D = Dummy()
        D.info = {
            'metric': 'bac_metric',
            'task': task_type,
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
            include_estimators=['ridge'],
            include_preprocessors=['select_rates'])
        configuration = configuration_space.sample_configuration()

        evaluator = HoldoutEvaluator(D, configuration)
        pred = evaluator.predict_proba(None, model, task_type)
        expected = [[0.9], [0.3]]
        for i in range(len(expected)):
            self.assertEqual(expected[i], pred[i])


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.test_evaluate']
    unittest.main()
