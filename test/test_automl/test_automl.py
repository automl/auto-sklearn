# -*- encoding: utf-8 -*-
import gzip
import os
import pickle
import sys
import time
import unittest
import unittest.mock

import numpy as np
import sklearn.datasets

from autosklearn.util.backend import Backend, BackendContext
from autosklearn.automl import AutoML
import autosklearn.automl
import autosklearn.pipeline.util as putil
from autosklearn.util import setup_logger, get_logger, backend
from autosklearn.constants import *
from autosklearn.smbo import load_data

sys.path.append(os.path.dirname(__file__))
from base import Base


class AutoMLStub(AutoML):
    def __init__(self):
        self.__class__ = AutoML


class AutoMLTest(Base, unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()

        self.automl = AutoMLStub()

        self.automl._shared_mode = False
        self.automl._seed = 42
        self.automl._backend = unittest.mock.Mock(spec=Backend)
        self.automl._delete_output_directories = lambda: 0

    def test_refit_shuffle_on_fail(self):
        output = os.path.join(self.test_dir, '..', '.tmp_refit_shuffle_on_fail')
        context = BackendContext(output, output, False, False)
        backend = Backend(context)

        failing_model = unittest.mock.Mock()
        failing_model.fit.side_effect = [ValueError(), ValueError(), None]

        auto = AutoML(backend, 30, 5)
        ensemble_mock = unittest.mock.Mock()
        auto.ensemble_ = ensemble_mock
        ensemble_mock.get_model_identifiers.return_value = [1]

        auto.models_ = {1: failing_model}

        X = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        auto.refit(X, y)

        self.assertEqual(failing_model.fit.call_count, 3)

    def test_only_loads_ensemble_models(self):
        identifiers = [(1, 2), (3, 4)]

        models = [42]
        self.automl._backend.load_ensemble.return_value.identifiers_ \
            = identifiers
        self.automl._backend.load_models_by_identifiers.side_effect \
            = lambda ids: models if ids is identifiers else None

        self.automl._load_models()

        self.assertEqual(models, self.automl.models_)

    def test_loads_all_models_if_no_ensemble(self):
        models = [42]
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.load_all_models.return_value = models

        self.automl._load_models()

        self.assertEqual(models, self.automl.models_)

    def test_raises_if_no_models(self):
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.load_all_models.return_value = []
        self.automl._resampling_strategy = 'holdout'

        self.assertRaises(ValueError, self.automl._load_models)

    def test_fit(self):
        output = os.path.join(self.test_dir, '..', '.tmp_test_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        backend_api = backend.create(output, output)
        automl = autosklearn.automl.AutoML(backend_api, 30, 5)
        automl.fit(X_train, Y_train)
        #print(automl.show_models(), flush=True)
        #print(automl.cv_results_, flush=True)
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

    def test_binary_score_and_include(self):
        """
        Test fix for binary classification prediction
        taking the index 1 of second dimension in prediction matrix
        """

        output = os.path.join(self.test_dir, '..', '.tmp_test_binary_score')
        self._setUp(output)

        data = sklearn.datasets.make_classification(
            n_samples=400, n_features=10, n_redundant=1, n_informative=3,
            n_repeated=1, n_clusters_per_class=2, random_state=1)
        X_train = data[0][:200]
        Y_train = data[1][:200]
        X_test = data[0][200:]
        Y_test = data[1][200:]

        backend_api = backend.create(output, output)
        automl = autosklearn.automl.AutoML(backend_api, 30, 5,
                                           include_estimators=['sgd'],
                                           include_preprocessors=['no_preprocessing'])
        automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION)
        #print(automl.show_models(), flush=True)
        #print(automl.cv_results_, flush=True)
        self.assertEqual(automl._task, BINARY_CLASSIFICATION)

        # TODO, the assumption from above is not really tested here
        # Also, the score method should be removed, it only makes little sense
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.4)

        del automl
        self._tearDown(output)

    def test_automl_outputs(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_automl_outputs')
        self._setUp(output)
        name = '31_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        data_manager_file = os.path.join(output, '.auto-sklearn',
                                         'datamanager.pkl.gz')

        backend_api = backend.create(output, output)
        auto = autosklearn.automl.AutoML(
            backend_api, 30, 5,
            initial_configurations_via_metalearning=25,
            seed=100)
        auto.fit_automl_dataset(dataset)

        # pickled data manager (without one hot encoding!)
        with gzip.open(data_manager_file, 'rb') as fh:
            D = pickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'][0, :3],
                                        [1., 12., 2.]))

        # Check that all directories are there
        fixture = ['predictions_valid', 'true_targets_ensemble.npy.gz',
                   'start_time_100', 'datamanager.pkl.gz',
                   'predictions_ensemble',
                   'ensembles', 'predictions_test', 'models']
        self.assertEqual(sorted(os.listdir(os.path.join(output,
                                                        '.auto-sklearn'))),
                         sorted(fixture))

        # At least one ensemble, one validation, one test prediction and one
        # model and one ensemble
        fixture = os.listdir(os.path.join(output, '.auto-sklearn',
                                          'predictions_ensemble'))
        self.assertIn('predictions_ensemble_100_00001.npy.gz', fixture)

        fixture = os.listdir(os.path.join(output, '.auto-sklearn',
                                          'models'))
        self.assertIn('100.1.model.gz', fixture)

        fixture = os.listdir(os.path.join(output, '.auto-sklearn',
                                          'ensembles'))
        self.assertIn('100.0000000000.ensemble', fixture)

        # Start time
        start_time_file_path = os.path.join(output, '.auto-sklearn',
                                            "start_time_100")
        with open(start_time_file_path, 'r') as fh:
            start_time = float(fh.read())
        self.assertGreaterEqual(time.time() - start_time, 10)

        del auto
        self._tearDown(output)

    def test_do_dummy_prediction(self):
        for name in ['401_bac', '31_bac', 'adult', 'cadata']:
            output = os.path.join(self.test_dir, '..',
                                  '.tmp_test_do_dummy_prediction')
            self._setUp(output)

            dataset = os.path.join(self.test_dir, '..', '.data', name)

            backend_api = backend.create(output, output)
            auto = autosklearn.automl.AutoML(
                backend_api, 30, 5,
                initial_configurations_via_metalearning=25)
            setup_logger()
            auto._logger = get_logger('test_do_dummy_predictions')
            auto._backend._make_internals_directory()
            D = load_data(dataset, backend_api)
            auto._backend.save_datamanager(D)
            auto._do_dummy_prediction(D, 1)

            # Ensure that the dummy predictions are not in the current working
            # directory, but in the output directory (under output)
            self.assertFalse(os.path.exists(os.path.join(os.getcwd(),
                                                         '.auto-sklearn')))
            self.assertTrue(os.path.exists(os.path.join(
                output, '.auto-sklearn', 'predictions_ensemble',
                'predictions_ensemble_1_00001.npy.gz')))

            del auto
            self._tearDown(output)


