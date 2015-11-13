# -*- encoding: utf-8 -*-
import os

import numpy as np
import ParamSklearn.util as putil

import autosklearn
from autosklearn.util.backend import Backend
from autosklearn.constants import *
from base import Base


class ArrayReturningDummyPredictor(object):
    def __init__(self, test):
        self.arr = test

    def predict_proba(self, X):
        return self.arr

class EstimatorTest(Base):
    _multiprocess_can_split_ = True

    def test_fit(self):
        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=12,
                                                   per_run_time_limit=12,
                                                   tmp_folder=output,
                                                   output_folder=output)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        print(automl.show_models())

        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

    def test_pSMAC_wrong_arguments(self):
        self.assertRaisesRegexp(ValueError,
                                "If shared_mode == True tmp_folder must not "
                                "be None.",
                                autosklearn.AutoSklearnClassifier,
                                shared_mode=True)

        self.assertRaisesRegexp(ValueError,
                                "If shared_mode == True output_folder must not "
                                "be None.",
                                autosklearn.AutoSklearnClassifier,
                                shared_mode=True,
                                tmp_folder='/tmp/duitaredxtvbedb')

    def test_fit_pSMAC(self):
        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_pSMAC')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

        automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=15,
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

        automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=10,
                                                   per_run_time_limit=10,
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
                                                     'ensemble_indices'))), 1)
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

