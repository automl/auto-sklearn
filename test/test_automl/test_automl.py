# -*- encoding: utf-8 -*-
import os
import pickle
import sys
import time
import glob
import unittest
import unittest.mock

import numpy as np
import sklearn.datasets
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR

from autosklearn.util.backend import Backend
from autosklearn.automl import AutoML
import autosklearn.automl
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.metrics import accuracy
import autosklearn.pipeline.util as putil
from autosklearn.util.logging_ import setup_logger, get_logger
from autosklearn.constants import MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION, REGRESSION
from smac.tae.execute_ta_run import StatusType

sys.path.append(os.path.dirname(__file__))
from base import Base  # noqa (E402: module level import not at top of file)


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
        failing_model.fit_transformer.side_effect = [
            ValueError(), ValueError(), (None, {})]
        failing_model.get_max_iter.return_value = 100

        auto = AutoML(backend_api, 20, 5)
        ensemble_mock = unittest.mock.Mock()
        ensemble_mock.get_selected_model_identifiers.return_value = [(1, 1, 50.0)]
        auto.ensemble_ = ensemble_mock
        for budget_type in [None, 'iterations']:
            auto._budget_type = budget_type

            auto.models_ = {(1, 1, 50.0): failing_model}

            X = np.array([1, 2, 3])
            y = np.array([1, 2, 3])
            auto.refit(X, y)

            self.assertEqual(failing_model.fit.call_count, 3)
        self.assertEqual(failing_model.fit_transformer.call_count, 3)

        del auto
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_only_loads_ensemble_models(self):

        def side_effect(ids, *args, **kwargs):
            return models if ids is identifiers else {}

        # Add a resampling strategy as this is required by load_models
        self.automl._resampling_strategy = 'holdout'
        identifiers = [(1, 2), (3, 4)]

        models = [42]
        load_ensemble_mock = unittest.mock.Mock()
        load_ensemble_mock.get_selected_model_identifiers.return_value = identifiers
        self.automl._backend.load_ensemble.return_value = load_ensemble_mock
        self.automl._backend.load_models_by_identifiers.side_effect = side_effect

        self.automl._load_models()
        self.assertEqual(models, self.automl.models_)
        self.assertIsNone(self.automl.cv_models_)

        self.automl._resampling_strategy = 'cv'

        models = [42]
        self.automl._backend.load_cv_models_by_identifiers.side_effect = side_effect

        self.automl._load_models()
        self.assertEqual(models, self.automl.cv_models_)

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
        automl = autosklearn.automl.AutoML(
            backend=backend_api,
            time_left_for_this_task=20,
            per_run_time_limit=5,
            metric=accuracy,
        )
        automl.fit(
            X_train, Y_train, task=MULTICLASS_CLASSIFICATION,
        )
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_delete_non_candidate_models(self):
        backend_api = self._create_backend(
            'test_delete', delete_tmp_folder_after_terminate=False)

        seed = 555
        X, Y, _, _ = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(
            backend_api,
            time_left_for_this_task=30,
            per_run_time_limit=5,
            ensemble_nbest=3,
            seed=seed,
            initial_configurations_via_metalearning=0,
            resampling_strategy='holdout',
            include_estimators=['sgd'],
            include_preprocessors=['no_preprocessing'],
            metric=accuracy,
        )

        automl.fit(X, Y, task=MULTICLASS_CLASSIFICATION,
                   X_test=X, y_test=Y)

        # Assert at least one model file has been deleted and that there were no
        # deletion errors
        log_file_path = glob.glob(os.path.join(
            backend_api.temporary_directory, 'AutoML(' + str(seed) + '):*.log'))
        with open(log_file_path[0]) as log_file:
            log_content = log_file.read()
            self.assertIn('Deleted files of non-candidate model', log_content)
            self.assertNotIn('Failed to delete files of non-candidate model', log_content)
            self.assertNotIn('Failed to lock model', log_content)

        # Assert that the files of the models used by the ensemble weren't deleted
        model_files = backend_api.list_all_models(seed=seed)
        model_files_idx = set()
        for m_file in model_files:
            # Extract the model identifiers from the filename
            m_file = os.path.split(m_file)[1].replace('.model', '').split('.', 2)
            model_files_idx.add((int(m_file[0]), int(m_file[1]), float(m_file[2])))
        ensemble_members_idx = set(automl.ensemble_.identifiers_)
        self.assertTrue(ensemble_members_idx.issubset(model_files_idx))

        del automl
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_fit_roar(self):
        def get_roar_object_callback(
                scenario_dict,
                seed,
                ta,
                ta_kwargs,
                **kwargs
        ):
            """Random online adaptive racing.

            http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf"""
            scenario = Scenario(scenario_dict)
            return ROAR(
                scenario=scenario,
                rng=seed,
                tae_runner=ta,
                tae_runner_kwargs=ta_kwargs,
            )

        backend_api = self._create_backend('test_fit_roar')

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(
            backend=backend_api,
            time_left_for_this_task=20,
            per_run_time_limit=5,
            initial_configurations_via_metalearning=0,
            get_smac_object_callback=get_roar_object_callback,
            metric=accuracy,
        )
        setup_logger()
        automl._logger = get_logger('test_fit_roar')
        automl.fit(
            X_train, Y_train, task=MULTICLASS_CLASSIFICATION,
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

        automl = autosklearn.automl.AutoML(
            backend_api, 20, 5,
            include_estimators=['sgd'],
            include_preprocessors=['no_preprocessing'],
            metric=accuracy,
        )
        automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION)
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

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        name = 'iris'
        data_manager_file = os.path.join(
            backend_api.temporary_directory,
            '.auto-sklearn',
            'datamanager.pkl'
        )

        auto = autosklearn.automl.AutoML(
            backend_api, 20, 5,
            initial_configurations_via_metalearning=0,
            seed=100,
            metric=accuracy,
        )
        setup_logger()
        auto._logger = get_logger('test_automl_outputs')
        auto.fit(
            X=X_train,
            y=Y_train,
            X_test=X_test,
            y_test=Y_test,
            dataset_name=name,
            task=MULTICLASS_CLASSIFICATION,
        )

        # pickled data manager (without one hot encoding!)
        with open(data_manager_file, 'rb') as fh:
            D = pickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'], X_train))

        # Check that all directories are there
        fixture = ['cv_models', 'true_targets_ensemble.npy',
                   'start_time_100', 'datamanager.pkl',
                   'predictions_ensemble',
                   'ensembles', 'predictions_test', 'models']
        self.assertEqual(sorted(os.listdir(os.path.join(backend_api.temporary_directory,
                                                        '.auto-sklearn'))),
                         sorted(fixture))

        # At least one ensemble, one validation, one test prediction and one
        # model and one ensemble
        fixture = os.listdir(os.path.join(backend_api.temporary_directory,
                                          '.auto-sklearn', 'predictions_ensemble'))
        self.assertGreater(len(fixture), 0)

        fixture = glob.glob(os.path.join(backend_api.temporary_directory, '.auto-sklearn',
                                         'models', '100.*.model'))
        self.assertGreater(len(fixture), 0)

        fixture = os.listdir(os.path.join(backend_api.temporary_directory,
                                          '.auto-sklearn', 'ensembles'))
        self.assertIn('100.0000000001.ensemble', fixture)

        # Start time
        start_time_file_path = os.path.join(backend_api.temporary_directory,
                                            '.auto-sklearn', "start_time_100")
        with open(start_time_file_path, 'r') as fh:
            start_time = float(fh.read())
        self.assertGreaterEqual(time.time() - start_time, 10)

        del auto
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    def test_do_dummy_prediction(self):
        datasets = {
            'breast_cancer': BINARY_CLASSIFICATION,
            'wine': MULTICLASS_CLASSIFICATION,
            'diabetes': REGRESSION,
        }

        for name, task in datasets.items():
            backend_api = self._create_backend('test_do_dummy_prediction')

            X_train, Y_train, X_test, Y_test = putil.get_dataset(name)
            datamanager = XYDataManager(
                X_train, Y_train,
                X_test, Y_test,
                task=task,
                dataset_name=name,
                feat_type=None,
            )

            auto = autosklearn.automl.AutoML(
                backend_api, 20, 5,
                initial_configurations_via_metalearning=25,
                metric=accuracy,
            )
            setup_logger()
            auto._logger = get_logger('test_do_dummy_predictions')
            auto._backend.save_datamanager(datamanager)
            D = backend_api.load_datamanager()

            # Check if data manager is correcly loaded
            self.assertEqual(D.info['task'], datamanager.info['task'])

            auto._do_dummy_prediction(D, 1)

            # Ensure that the dummy predictions are not in the current working
            # directory, but in the temporary directory.
            self.assertFalse(os.path.exists(os.path.join(os.getcwd(),
                                                         '.auto-sklearn')))
            self.assertTrue(os.path.exists(os.path.join(
                backend_api.temporary_directory, '.auto-sklearn', 'predictions_ensemble',
                'predictions_ensemble_1_1_0.0.npy')))

            del auto
            self._tearDown(backend_api.temporary_directory)
            self._tearDown(backend_api.output_directory)

    @unittest.mock.patch('autosklearn.evaluation.ExecuteTaFuncWithQueue.run')
    def test_fail_if_dummy_prediction_fails(self, ta_run_mock):
        backend_api = self._create_backend('test_fail_if_dummy_prediction_fails')

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        datamanager = XYDataManager(
            X_train, Y_train,
            X_test, Y_test,
            task=2,
            feat_type=['Numerical' for i in range(X_train.shape[1])],
            dataset_name='iris',
        )

        time_for_this_task = 30
        per_run_time = 10
        auto = autosklearn.automl.AutoML(backend_api,
                                         time_for_this_task,
                                         per_run_time,
                                         initial_configurations_via_metalearning=25,
                                         metric=accuracy,
                                         )
        setup_logger()
        auto._logger = get_logger('test_fail_if_dummy_prediction_fails')
        auto._backend._make_internals_directory()
        auto._backend.save_datamanager(datamanager)

        # First of all, check that ta.run() is actually called.
        ta_run_mock.return_value = StatusType.SUCCESS, None, None, "test"
        auto._do_dummy_prediction(datamanager, 1)
        ta_run_mock.assert_called_once_with(1, cutoff=time_for_this_task)

        # Case 1. Check that function raises no error when statustype == success.
        # ta.run() returns status, cost, runtime, and additional info.
        ta_run_mock.return_value = StatusType.SUCCESS, None, None, "test"
        raised = False
        try:
            auto._do_dummy_prediction(datamanager, 1)
        except ValueError:
            raised = True
        self.assertFalse(raised, 'Exception raised')

        # Case 2. Check that if statustype returned by ta.run() != success,
        # the function raises error.
        ta_run_mock.return_value = StatusType.CRASHED, None, None, "test"
        self.assertRaisesRegex(ValueError,
                               'Dummy prediction failed with run state StatusType.CRASHED '
                               'and additional output: test.',
                               auto._do_dummy_prediction,
                               datamanager, 1,
                               )
        ta_run_mock.return_value = StatusType.ABORT, None, None, "test"
        self.assertRaisesRegex(ValueError,
                               'Dummy prediction failed with run state StatusType.ABORT '
                               'and additional output: test.',
                               auto._do_dummy_prediction,
                               datamanager, 1,
                               )
        ta_run_mock.return_value = StatusType.TIMEOUT, None, None, "test"
        self.assertRaisesRegex(ValueError,
                               'Dummy prediction failed with run state StatusType.TIMEOUT '
                               'and additional output: test.',
                               auto._do_dummy_prediction,
                               datamanager, 1,
                               )
        ta_run_mock.return_value = StatusType.MEMOUT, None, None, "test"
        self.assertRaisesRegex(ValueError,
                               'Dummy prediction failed with run state StatusType.MEMOUT '
                               'and additional output: test.',
                               auto._do_dummy_prediction,
                               datamanager, 1,
                               )
        ta_run_mock.return_value = StatusType.CAPPED, None, None, "test"
        self.assertRaisesRegex(ValueError,
                               'Dummy prediction failed with run state StatusType.CAPPED '
                               'and additional output: test.',
                               auto._do_dummy_prediction,
                               datamanager, 1,
                               )

        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

    @unittest.mock.patch('autosklearn.smbo.AutoMLSMBO.run_smbo')
    def test_exceptions_inside_log_in_smbo(self, smbo_run_mock):

        # Make sure that any exception during the AutoML fit due to
        # SMAC are properly captured in a log file
        backend_api = self._create_backend('test_exceptions_inside_log')
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)

        automl = autosklearn.automl.AutoML(
            backend_api,
            20,
            5,
            metric=accuracy,
        )

        output_file = 'test_exceptions_inside_log.log'
        setup_logger(output_file=output_file)
        logger = get_logger('test_exceptions_inside_log')

        # Create a custom exception to prevent other errors to slip in
        class MyException(Exception):
            pass

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        # The first call is on dummy predictor failure
        message = str(np.random.randint(100)) + '_run_smbo'
        smbo_run_mock.side_effect = MyException(message)

        with unittest.mock.patch('autosklearn.automl.AutoML._get_logger') as mock:
            mock.return_value = logger
            with self.assertRaises(MyException):
                automl.fit(
                    X_train,
                    Y_train,
                    task=MULTICLASS_CLASSIFICATION,
                )
            with open(output_file) as f:
                self.assertTrue(message in f.read())

        # Cleanup
        os.unlink(output_file)
        self._tearDown(backend_api.temporary_directory)
        self._tearDown(backend_api.output_directory)


if __name__ == "__main__":
    unittest.main()
