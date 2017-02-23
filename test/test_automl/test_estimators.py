import collections
import gzip
import os
import pickle
import sys
import unittest
import unittest.mock

import sklearn

import numpy as np
import numpy.ma as npma
from sklearn.grid_search import _CVScoreTuple

import autosklearn.pipeline.util as putil
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.estimators import AutoMLClassifier
from autosklearn.util.backend import Backend, BackendContext
from autosklearn.constants import *
sys.path.append(os.path.dirname(__file__))
from base import Base


class ArrayReturningDummyPredictor(object):
    def __init__(self, test):
        self.arr = test

    def predict_proba(self, X):
        return self.arr

class EstimatorTest(Base, unittest.TestCase):
    _multiprocess_can_split_ = True

    # def test_fit_partial_cv(self):
    #
    #     output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_partial_cv')
    #     self._setUp(output)
    #
    #     X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    #     automl = AutoSklearnClassifier(time_left_for_this_task=30,
    #                                    per_run_time_limit=5,
    #                                    tmp_folder=output,
    #                                    output_folder=output,
    #                                    resampling_strategy='partial-cv',
    #                                    ensemble_size=0,
    #                                    delete_tmp_folder_after_terminate=False)
    #     automl.fit(X_train, Y_train)
    #
    #     del automl
    #     self._tearDown(output)

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

    def test_fit_pSMAC(self):
        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_pSMAC')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

        # test parallel Classifier to predict classes, not only indexes
        Y_train = Y_train + 1
        Y_test = Y_test + 1

        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
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
        with open(true_targets_ensemble_path, 'rb') as fh:
            true_targets_ensemble = np.load(fh)
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
            probas_test[i, value - 1] = 1.0

        dummy = ArrayReturningDummyPredictor(probas_test)
        context = BackendContext(output, output, False, False)
        backend = Backend(context)
        backend.save_model(dummy, 30, 1)

        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
                                       output_folder=output,
                                       tmp_folder=output,
                                       shared_mode=True,
                                       seed=2,
                                       initial_configurations_via_metalearning=0,
                                       ensemble_size=0)
        automl.fit_ensemble(Y_train,
                            task=MULTICLASS_CLASSIFICATION,
                            metric=ACC_METRIC,
                            precision='32',
                            dataset_name='iris',
                            ensemble_size=20,
                            ensemble_nbest=50)
        #print(automl.show_models(), flush=True)

        predictions = automl.predict(X_test)
        score = sklearn.metrics.accuracy_score(Y_test, predictions)

        self.assertEqual(len(os.listdir(os.path.join(output, '.auto-sklearn',
                                                     'ensembles'))), 1)
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(automl._automl._automl._task, MULTICLASS_CLASSIFICATION)

        models = automl._automl._automl.models_
        classifier_types = [type(c) for c in models.values()]
        self.assertIn(ArrayReturningDummyPredictor, classifier_types)

        del automl
        self._tearDown(output)

    def test_grid_scores(self):
        output = os.path.join(self.test_dir, '..', '.tmp_grid_scores')
        self._setUp(output)

        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                    per_run_time_limit=5,
                                    output_folder=output,
                                    tmp_folder=output,
                                    shared_mode=False,
                                    seed=1,
                                    initial_configurations_via_metalearning=0,
                                    ensemble_size=0)
        cls_ = cls.build_automl()
        automl = cls_._automl
        automl.runhistory_ = unittest.mock.MagicMock()

        RunKey = collections.namedtuple(
            'RunKey', ['config_id', 'instance_id', 'seed'])

        RunValue = collections.namedtuple(
            'RunValue', ['cost', 'time', 'status', 'additional_info'])

        runhistory = dict()
        runhistory[RunKey(1, 1, 1)] = RunValue(1, 1, 1, '')
        automl.runhistory_.data = runhistory
        grid_scores_ = automl.grid_scores_

        self.assertIsInstance(grid_scores_[0], _CVScoreTuple)
        # In the runhistory we store losses, thus the score is zero
        self.assertEqual(grid_scores_[0].mean_validation_score, 0)
        self.assertEqual(grid_scores_[0].cv_validation_scores, [0])
        self.assertIsInstance(grid_scores_[0].parameters, unittest.mock.MagicMock)

        del automl
        self._tearDown(output)

    def test_cv_results(self):
        # TODO restructure and actually use real SMAC output from a long run
        # to do this unittest!
        output = os.path.join(self.test_dir, '..', '.tmp_cv_results')
        self._setUp(output)
        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                    per_run_time_limit=5,
                                    output_folder=output,
                                    tmp_folder=output,
                                    shared_mode=False,
                                    seed=1,
                                    initial_configurations_via_metalearning=0,
                                    ensemble_size=0)
        cls.fit(X_train, Y_train)
        cv_results = cls.cv_results_
        self.assertIsInstance(cv_results, dict)
        self.assertIsInstance(cv_results['mean_test_score'], np.ndarray)
        self.assertIsInstance(cv_results['mean_fit_time'], np.ndarray)
        self.assertIsInstance(cv_results['params'], list)
        self.assertIsInstance(cv_results['rank_test_scores'], np.ndarray)
        self.assertTrue([isinstance(val, npma.MaskedArray) for key, val in
                         cv_results.items() if key.startswith('param_')])
        del cls
        self._tearDown(output)

        
class AutoMLClassifierTest(Base, unittest.TestCase):

    def test_multiclass_prediction(self):
        classes = [['a', 'b', 'c']]
        predicted_probabilities = [[0, 0, 0.99], [0, 0.99, 0], [0.99, 0, 0],
                                   [0, 0.99, 0], [0, 0, 0.99]]
        predicted_indexes = [2, 1, 0, 1, 2]
        expected_result = ['c', 'b', 'a', 'b', 'c']

        automl_mock = unittest.mock.Mock()
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

        automl_mock = unittest.mock.Mock()
        automl_mock.predict.return_value = np.matrix(predicted_probabilities)

        classifier = AutoMLClassifier(automl_mock)
        classifier._classes = list(map(np.array, classes))
        classifier._n_outputs = 2
        classifier._n_classes = np.array([3, 2])

        actual_result = classifier.predict([None] * len(predicted_indexes))

        np.testing.assert_array_equal(expected_result, actual_result)

    def test_can_pickle_classifier(self):
        output = os.path.join(self.test_dir, '..', '.tmp_can_pickle')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
                                       tmp_folder=output,
                                       output_folder=output)
        automl.fit(X_train, Y_train)

        initial_predictions = automl.predict(X_test)
        initial_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                          initial_predictions)
        self.assertTrue(initial_accuracy > 0.75)

        # Test pickle
        dump_file = os.path.join(output, 'automl.dump.pkl')

        with open(dump_file, 'wb') as f:
            pickle.dump(automl, f)

        with open(dump_file, 'rb') as f:
            restored_automl = pickle.load(f)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                           restored_predictions)
        self.assertTrue(restored_accuracy > 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)

        # Test joblib
        dump_file = os.path.join(output, 'automl.dump.joblib')

        sklearn.externals.joblib.dump(automl, dump_file)

        restored_automl = sklearn.externals.joblib.load(dump_file)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                           restored_predictions)
        self.assertTrue(restored_accuracy > 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)
