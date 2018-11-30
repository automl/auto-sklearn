# -*- encoding: utf-8 -*-
import os
import pickle
import sys
import time
import unittest
import unittest.mock

import numpy as np
import sklearn.datasets
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR

from autosklearn.util.backend import Backend, BackendContext
from autosklearn.automl import AutoML
import autosklearn.automl
from autosklearn.metrics import accuracy
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
        backend_api = self._create_backend('test_refit_shuffle_on_fail')

        failing_model = unittest.mock.Mock()
        failing_model.fit.side_effect = [ValueError(), ValueError(), None]

        auto = AutoML(backend_api, 20, 5)
        ensemble_mock = unittest.mock.Mock()
        auto.ensemble_ = ensemble_mock
        ensemble_mock.get_selected_model_identifiers.return_value = [1]

        auto.models_ = {1: failing_model}

        X = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        auto.refit(X, y)

        self.assertEqual(failing_model.fit.call_count, 3)

        del auto
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_only_loads_ensemble_models(self):
        identifiers = [(1, 2), (3, 4)]

        models = [42]
        self.automl._backend.load_ensemble.return_value.identifiers_ \
            = identifiers
        self.automl._backend.load_models_by_identifiers.side_effect \
            = lambda ids: models if ids is identifiers else None

        self.automl._load_models()

        self.assertEqual(models, self.automl.models_)

    def test_check_for_models_if_no_ensemble(self):
        models = [42]
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.list_all_models.return_value = models
        self.automl._disable_evaluator_output = False

        self.automl._load_models()

    def test_raises_if_no_models(self):
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.list_all_models.return_value = []
        self.automl._resampling_strategy = 'holdout'

        self.automl._disable_evaluator_output = False
        self.assertRaises(ValueError, self.automl._load_models)

        self.automl._disable_evaluator_output = True
        self.automl._load_models()

    def test_fit(self):
        backend_api = self._create_backend('test_fit')

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(backend_api, 20, 5)
        automl.fit(
            X_train, Y_train, metric=accuracy, task=MULTICLASS_CLASSIFICATION,
        )
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_fit_roar(self):
        def get_roar_object_callback(
                scenario_dict,
                seed,
                ta,
                **kwargs
        ):
            """Random online adaptive racing.

            http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf"""
            scenario = Scenario(scenario_dict)
            return ROAR(
                scenario=scenario,
                rng=seed,
                tae_runner=ta,
            )

        backend_api = self._create_backend('test_fit_roar')

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(
            backend=backend_api,
            time_left_for_this_task=20,
            per_run_time_limit=5,
            initial_configurations_via_metalearning=0,
            get_smac_object_callback=get_roar_object_callback,
        )
        automl.fit(
            X_train, Y_train, metric=accuracy, task=MULTICLASS_CLASSIFICATION,
        )
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_binary_score_and_include(self):
        """
        Test fix for binary classification prediction
        taking the index 1 of second dimension in prediction matrix
        """
        backend_api = self._create_backend('test_binary_score_and_include')

        data = sklearn.datasets.make_classification(
            n_samples=400, n_features=10, n_redundant=1, n_informative=3,
            n_repeated=1, n_clusters_per_class=2, random_state=1)
        X_train = data[0][:200]
        Y_train = data[1][:200]
        X_test = data[0][200:]
        Y_test = data[1][200:]

        automl = autosklearn.automl.AutoML(backend_api, 20, 5,
                                           include_estimators=['sgd'],
                                           include_preprocessors=['no_preprocessing'])
        automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION,
                   metric=accuracy)
        self.assertEqual(automl._task, BINARY_CLASSIFICATION)

        # TODO, the assumption from above is not really tested here
        # Also, the score method should be removed, it only makes little sense
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.4)

        del automl
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_automl_outputs(self):
        backend_api = self._create_backend('test_automl_outputs')

        name = '31_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        data_manager_file = os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                         'datamanager.pkl')

        auto = autosklearn.automl.AutoML(
            backend_api, 20, 5,
            initial_configurations_via_metalearning=0,
            seed=100,
        )
        auto.fit_automl_dataset(dataset, accuracy)

        # pickled data manager (without one hot encoding!)
        with open(data_manager_file, 'rb') as fh:
            D = pickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'][0, :3],
                                        [1., 12., 2.]))

        # Check that all directories are there
        fixture = ['predictions_valid', 'true_targets_ensemble.npy',
                   'start_time_100', 'datamanager.pkl',
                   'predictions_ensemble',
                   'ensembles', 'predictions_test', 'models']
        self.assertEqual(sorted(os.listdir(os.path.join(backend_api.temporary_directory,
                                                        '.auto-sklearn'))),
                         sorted(fixture))

        # At least one ensemble, one validation, one test prediction and one
        # model and one ensemble
        fixture = os.listdir(os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                          'predictions_ensemble'))
        self.assertIn('predictions_ensemble_100_1.npy', fixture)

        fixture = os.listdir(os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                          'models'))
        self.assertIn('100.1.model', fixture)

        fixture = os.listdir(os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                          'ensembles'))
        self.assertIn('100.0000000000.ensemble', fixture)

        # Start time
        start_time_file_path = os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                            "start_time_100")
        with open(start_time_file_path, 'r') as fh:
            start_time = float(fh.read())
        self.assertGreaterEqual(time.time() - start_time, 10)

        del auto
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_do_dummy_prediction(self):
        for name in ['401_bac', '31_bac', 'adult', 'cadata']:
            backend_api = self._create_backend('test_do_dummy_prediction')

            dataset = os.path.join(self.test_dir, '..', '.data', name)

            auto = autosklearn.automl.AutoML(
                backend_api, 20, 5,
                initial_configurations_via_metalearning=25)
            setup_logger()
            auto._logger = get_logger('test_do_dummy_predictions')
            auto._backend._make_internals_directory()
            D = load_data(dataset, backend_api)
            auto._backend.save_datamanager(D)
            auto._do_dummy_prediction(D, 1)

            # Ensure that the dummy predictions are not in the current working
            # directory, but in the temporary directory.
            self.assertFalse(os.path.exists(os.path.join(os.getcwd(),
                                                         '.auto-sklearn')))
            self.assertTrue(os.path.exists(os.path.join(
                backend_api.temporary_directory, '.auto-sklearn', 'predictions_ensemble',
                'predictions_ensemble_1_1.npy')))

            del auto
            self._tearDown(backend_api.temporary_directory)
            self._tearDown(backend_api.output_directory)


if __name__=="__main__":
    unittest.main()
