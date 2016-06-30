# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import sys
import unittest
import mock

import numpy as np
import autosklearn.pipeline.util as putil

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.estimators import AutoMLClassifier
from autosklearn.util.backend import Backend
from autosklearn.constants import *
from autosklearn.automl import AutoML

sys.path.append(os.path.dirname(__file__))
from base import Base


class ArrayReturningDummyPredictor(object):
    def __init__(self, test):
        self.arr = test

    def predict_proba(self, X):
        return self.arr

class EstimatorTest(Base, unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_fit(self):
        if self.travis:
            self.skipTest('This test does currently not run on travis-ci. '
                          'Make sure it runs locally on your machine!')

        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                       per_run_time_limit=15,
                                       tmp_folder=output,
                                       output_folder=output)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        print(automl.show_models())

        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._automl._automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

    def test_pSMAC_wrong_arguments(self):
        self.assertRaisesRegexp(ValueError,
                                "If shared_mode == True tmp_folder must not "
                                "be None.",
                                lambda shared_mode: AutoSklearnClassifier(shared_mode=shared_mode).fit(None, None),
                                shared_mode=True)

        self.assertRaisesRegexp(ValueError,
                                "If shared_mode == True output_folder must not "
                                "be None.",
                                lambda shared_mode, tmp_folder:
                                AutoSklearnClassifier(shared_mode=shared_mode, tmp_folder=tmp_folder).fit(None, None),
                                shared_mode=True,
                                tmp_folder='/tmp/duitaredxtvbedb')

    def test_feat_type_wrong_arguments(self):
        cls = AutoSklearnClassifier()
        X = np.zeros((100, 100))
        y = np.zeros((100, ))
        self.assertRaisesRegexp(ValueError,
                                'Array feat_type does not have same number of '
                                'variables as X has features. 1 vs 100.',
                                cls.fit,
                                X=X, y=y, feat_type=[True])

        self.assertRaisesRegexp(ValueError,
                                'Array feat_type must only contain strings.',
                                cls.fit,
                                X=X, y=y, feat_type=[True]*100)

        self.assertRaisesRegexp(ValueError,
                                'Only `Categorical` and `Numerical` are '
                                'valid feature types, you passed `Car`',
                                cls.fit,
                                X=X, y=y, feat_type=['Car']*100)

    @unittest.skip("pSMAC not yet working with new python SMAC")
    def test_fit_pSMAC(self):
        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_pSMAC')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

        automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                       per_run_time_limit=15,
                                       output_folder=output,
                                       tmp_folder=output,
                                       shared_mode=True,
                                       seed=1,
                                       initial_configurations_via_metalearning=0,
                                       ensemble_size=0)
        automl.fit(X_train, Y_train)

        # Create a 'dummy model' for the first run, which has an accuracy of
        # more than 99%; it should be in the final ensemble if the ensemble
        # building of the second AutoSklearn classifier works correct
        true_targets_ensemble_path = os.path.join(output, '.auto-sklearn',
                                                  'true_targets_ensemble.npy')
        true_targets_ensemble = np.load(true_targets_ensemble_path)
        true_targets_ensemble[-1] = 1 if true_targets_ensemble[-1] != 1 else 0
        probas = np.zeros((len(true_targets_ensemble), 3), dtype=float)
        for i, value in enumerate(true_targets_ensemble):
            probas[i, value] = 1.0
        dummy_predictions_path = os.path.join(output, '.auto-sklearn',
                                              'predictions_ensemble',
                                              'predictions_ensemble_1_00030.npy')
        with open(dummy_predictions_path, 'wb') as fh:
            np.save(fh, probas)

        probas_test = np.zeros((len(Y_test), 3), dtype=float)
        for i, value in enumerate(Y_test):
            probas_test[i, value] = 1.0

        dummy = ArrayReturningDummyPredictor(probas_test)
        backend = Backend(output, output)
        backend.save_model(dummy, 30, 1)

        automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                       per_run_time_limit=15,
                                       output_folder=output,
                                       tmp_folder=output,
                                       shared_mode=True,
                                       seed=2,
                                       initial_configurations_via_metalearning=0,
                                       ensemble_size=0)
        automl.fit(X_train, Y_train)
        automl.run_ensemble_builder(0, 1, 50).wait()

        score = automl.score(X_test, Y_test)

        self.assertEqual(len(os.listdir(os.path.join(output, '.auto-sklearn',
                                                     'ensembles'))), 1)
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(automl._automl._automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)


class AutoMLClassifierTest(unittest.TestCase):

    def test_multiclass_prediction(self):
        classes = [['a', 'b', 'c']]
        predicted_probabilities = [[0, 0, 0.99], [0, 0.99, 0], [0.99, 0, 0],
                                   [0, 0.99, 0], [0, 0, 0.99]]
        predicted_indexes = [2, 1, 0, 1, 2]
        expected_result = ['c', 'b', 'a', 'b', 'c']

        automl_mock = mock.Mock()
        automl_mock.predict.return_value = np.array(predicted_probabilities)

        classifier = AutoMLClassifier(automl_mock)
        classifier._classes = [np.array(classes)]
        classifier._n_outputs = 1
        classifier._n_classes = np.array([3])

        actual_result = classifier.predict([None] * len(predicted_indexes))

        np.testing.assert_array_equal(expected_result, actual_result)

    def test_multilabel_prediction(self):
        classes = [['a', 'b', 'c'], [13, 17]]
        predicted_probabilities = [[[0, 0, 0.99], [0.99, 0]],
                                   [[0, 0.99, 0], [0.99, 0]],
                                   [[0.99, 0, 0], [0, 0.99]],
                                   [[0, 0.99, 0], [0, 0.99]],
                                   [[0, 0, 0.99], [0, 0.99]]]
        predicted_indexes = [[2, 0], [1, 0], [0, 1], [1, 1], [2, 1]]
        expected_result = np.array([['c', 13], ['b', 13], ['a', 17], ['b', 17], ['c', 17]], dtype=object)

        automl_mock = mock.Mock()
        automl_mock.predict.return_value = np.matrix(predicted_probabilities)

        classifier = AutoMLClassifier(automl_mock)
        classifier._classes = list(map(np.array, classes))
        classifier._n_outputs = 2
        classifier._n_classes = np.array([3, 2])

        actual_result = classifier.predict([None] * len(predicted_indexes))

        np.testing.assert_array_equal(expected_result, actual_result)
