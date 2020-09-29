import os
import sys
import time
import unittest.mock

import dask.distributed

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import RunValue, RunKey, RunHistory

from autosklearn.metrics import make_scorer, roc_auc, accuracy, log_loss
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.ensemble_builder import (
    EnsembleBuilder,
    ensemble_builder_process,
    Y_VALID,
    Y_TEST,
)
from autosklearn.ensembles.singlebest_ensemble import SingleBest

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)


def scorer_function(a, b):
    return 0.9


MockMetric = make_scorer('mock', scorer_function)


class BackendMock(object):

    def __init__(self):
        this_directory = os.path.abspath(
            os.path.dirname(__file__)
        )
        self.temporary_directory = os.path.join(
            this_directory, 'data',
        )

    def load_datamanager(self):
        manager = unittest.mock.Mock()
        manager.__reduce__ = lambda self: (unittest.mock.MagicMock, ())
        array = np.load(os.path.join(
            this_directory, 'data',
            '.auto-sklearn',
            'predictions_test',
            'predictions_test_0_3_100.0.npy'
        ))
        manager.data.get.return_value = array
        return manager

    def load_targets_ensemble(self):
        with open(os.path.join(
            self.temporary_directory,
            ".auto-sklearn",
            "predictions_ensemble",
            "predictions_ensemble_true.npy"
        ), "rb") as fp:
            y = np.load(fp, allow_pickle=True)
        return y

    def get_done_directory(self):
        return os.path.join(this_directory, 'data', '.auto-sklearn', 'done')

    def save_ensemble(self, ensemble, index_run, seed):
        return

    def save_predictions_as_txt(self, predictions, subset, idx, prefix, precision):
        return


class EnsembleBuilderMemMock(EnsembleBuilder):

    def fit_ensemble(self, selected_keys):
        np.ones([10000000, 1000000])


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self.backend = BackendMock()

    def tearDown(self):
        pass

    def testRead(self):

        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  # Binary Classification
            metric=roc_auc,
            seed=0,  # important to find the test files
        )

        success = ensbuilder.score_ensemble_preds()
        self.assertTrue(success, str(ensbuilder.read_preds))
        self.assertEqual(len(ensbuilder.read_preds), 3)

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
        )
        self.assertEqual(ensbuilder.read_preds[filename]["ens_score"], 0.5)

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
        )
        self.assertEqual(ensbuilder.read_preds[filename]["ens_score"], 1.0)

    def testNBest(self):
        for ensemble_nbest, models_on_disc, exp in (
                (1, None, 1),
                (1.0, None, 2),
                (0.1, None, 1),
                (0.9, None, 1),
                (1, 2, 1),
                (2, 1, 1),
        ):
            ensbuilder = EnsembleBuilder(
                backend=self.backend,
                dataset_name="TEST",
                task_type=1,  # Binary Classification
                metric=roc_auc,
                seed=0,  # important to find the test files
                ensemble_nbest=ensemble_nbest,
                max_models_on_disc=models_on_disc,
            )

            ensbuilder.score_ensemble_preds()
            sel_keys = ensbuilder.get_n_best_preds()

            self.assertEqual(len(sel_keys), exp)

            fixture = os.path.join(
                self.backend.temporary_directory,
                ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
            )
            self.assertEqual(sel_keys[0], fixture)

    def testMaxModelsOnDisc(self):

        ensemble_nbest = 4
        for (test_case, exp) in [
                # If None, no reduction
                (None, 2),
                # If Int, limit only on exceed
                (4, 2),
                (1, 1),
                # If Float, translate float to # models.
                # below, mock of each file is 100 Mb and
                # 4 files .model and .npy (test/val/pred) exist
                (700.0, 1),
                (800.0, 2),
                (9999.0, 2),
        ]:
            ensbuilder = EnsembleBuilder(
                backend=self.backend,
                dataset_name="TEST",
                task_type=1,  # Binary Classification
                metric=roc_auc,
                seed=0,  # important to find the test files
                ensemble_nbest=ensemble_nbest,
                max_models_on_disc=test_case,
            )

            with unittest.mock.patch('os.path.getsize') as mock:
                mock.return_value = 100*1024*1024
                ensbuilder.score_ensemble_preds()
                sel_keys = ensbuilder.get_n_best_preds()
                self.assertEqual(len(sel_keys), exp)

        # Test for Extreme scenarios
        # Make sure that the best predictions are kept
        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  # Binary Classification
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=50,
            max_models_on_disc=10000.0,
        )
        ensbuilder.read_preds = {}
        for i in range(50):
            ensbuilder.read_preds['pred'+str(i)] = {
                'ens_score': i*10,
                'num_run': i,
                0: True,
                'loaded': 1,
                "seed": 1,
                "disc_space_cost_mb": 50*i,
            }
        sel_keys = ensbuilder.get_n_best_preds()
        self.assertListEqual(['pred49', 'pred48', 'pred47', 'pred46'], sel_keys)

        # Make sure at least one model is kept alive
        ensbuilder.max_models_on_disc = 0.0
        sel_keys = ensbuilder.get_n_best_preds()
        self.assertListEqual(['pred49'], sel_keys)

    def testPerformanceRangeThreshold(self):
        to_test = ((0.0, 4), (0.1, 4), (0.3, 3), (0.5, 2), (0.6, 2), (0.8, 1),
                   (1.0, 1), (1, 1))
        for performance_range_threshold, exp in to_test:
            ensbuilder = EnsembleBuilder(
                backend=self.backend,
                dataset_name="TEST",
                task_type=1,  # Binary Classification
                metric=roc_auc,
                seed=0,  # important to find the test files
                ensemble_nbest=100,
                performance_range_threshold=performance_range_threshold
            )
            ensbuilder.read_preds = {
                'A': {'ens_score': 1, 'num_run': 1, 0: True, 'loaded': -1, "seed": 1},
                'B': {'ens_score': 2, 'num_run': 2, 0: True, 'loaded': -1, "seed": 1},
                'C': {'ens_score': 3, 'num_run': 3, 0: True, 'loaded': -1, "seed": 1},
                'D': {'ens_score': 4, 'num_run': 4, 0: True, 'loaded': -1, "seed": 1},
                'E': {'ens_score': 5, 'num_run': 5, 0: True, 'loaded': -1, "seed": 1},
            }
            sel_keys = ensbuilder.get_n_best_preds()

            self.assertEqual(len(sel_keys), exp)

    def testPerformanceRangeThresholdMaxBest(self):
        to_test = ((0.0, 1, 1), (0.0, 1.0, 4), (0.1, 2, 2), (0.3, 4, 3),
                   (0.5, 1, 1), (0.6, 10, 2), (0.8, 0.5, 1), (1, 1.0, 1))
        for performance_range_threshold, ensemble_nbest, exp in to_test:
            ensbuilder = EnsembleBuilder(
                backend=self.backend,
                dataset_name="TEST",
                task_type=1,  # Binary Classification
                metric=roc_auc,
                seed=0,  # important to find the test files
                ensemble_nbest=ensemble_nbest,
                performance_range_threshold=performance_range_threshold,
                max_models_on_disc=None,
            )
            ensbuilder.read_preds = {
                'A': {'ens_score': 1, 'num_run': 1, 0: True, 'loaded': -1, "seed": 1},
                'B': {'ens_score': 2, 'num_run': 2, 0: True, 'loaded': -1, "seed": 1},
                'C': {'ens_score': 3, 'num_run': 3, 0: True, 'loaded': -1, "seed": 1},
                'D': {'ens_score': 4, 'num_run': 4, 0: True, 'loaded': -1, "seed": 1},
                'E': {'ens_score': 5, 'num_run': 5, 0: True, 'loaded': -1, "seed": 1},
            }
            sel_keys = ensbuilder.get_n_best_preds()

            self.assertEqual(len(sel_keys), exp)

    def testFallBackNBest(self):

        ensbuilder = EnsembleBuilder(backend=self.backend,
                                     dataset_name="TEST",
                                     task_type=1,  # Binary Classification
                                     metric=roc_auc,
                                     seed=0,  # important to find the test files
                                     ensemble_nbest=1
                                     )

        ensbuilder.score_ensemble_preds()

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
        )
        ensbuilder.read_preds[filename]["ens_score"] = -1

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_3_100.0.npy"
        )
        ensbuilder.read_preds[filename]["ens_score"] = -1

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
        )
        ensbuilder.read_preds[filename]["ens_score"] = -1

        sel_keys = ensbuilder.get_n_best_preds()

        fixture = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
        )
        self.assertEqual(len(sel_keys), 1)
        self.assertEqual(sel_keys[0], fixture)

    def testGetValidTestPreds(self):

        ensbuilder = EnsembleBuilder(backend=self.backend,
                                     dataset_name="TEST",
                                     task_type=1,  # Binary Classification
                                     metric=roc_auc,
                                     seed=0,  # important to find the test files
                                     ensemble_nbest=1
                                     )

        ensbuilder.score_ensemble_preds()

        d1 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
        )
        d2 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
        )
        d3 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_3_100.0.npy"
        )

        sel_keys = ensbuilder.get_n_best_preds()
        self.assertEqual(len(sel_keys), 1)
        ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

        # Number of read files should be three and
        # predictions_ensemble_0_4_0.0.npy must not be in there
        self.assertEqual(len(ensbuilder.read_preds), 3)
        self.assertNotIn(
            os.path.join(
                self.backend.temporary_directory,
                ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_4_0.0.npy"
            ),
            ensbuilder.read_preds
        )

        # not selected --> should still be None
        self.assertIsNone(ensbuilder.read_preds[d1][Y_VALID])
        self.assertIsNone(ensbuilder.read_preds[d1][Y_TEST])
        self.assertIsNone(ensbuilder.read_preds[d3][Y_VALID])
        self.assertIsNone(ensbuilder.read_preds[d3][Y_TEST])

        # selected --> read valid and test predictions
        self.assertIsNotNone(ensbuilder.read_preds[d2][Y_VALID])
        self.assertIsNotNone(ensbuilder.read_preds[d2][Y_TEST])

    def testEntireEnsembleBuilder(self):

        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  # Binary Classification
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=2,
        )
        ensbuilder.SAVE2DISC = False

        ensbuilder.score_ensemble_preds()

        d2 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
        )

        sel_keys = ensbuilder.get_n_best_preds()
        self.assertGreater(len(sel_keys), 0)

        ensemble = ensbuilder.fit_ensemble(selected_keys=sel_keys)
        print(ensemble, sel_keys)

        n_sel_valid, n_sel_test = ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

        # both valid and test prediction files are available
        self.assertGreater(len(n_sel_valid), 0)
        self.assertEqual(n_sel_valid, n_sel_test)

        y_valid = ensbuilder.predict(
            set_="valid",
            ensemble=ensemble,
            selected_keys=n_sel_valid,
            n_preds=len(sel_keys),
            index_run=1,
        )
        y_test = ensbuilder.predict(
            set_="test",
            ensemble=ensemble,
            selected_keys=n_sel_test,
            n_preds=len(sel_keys),
            index_run=1,
        )

        # predictions for valid and test are the same
        # --> should results in the same predictions
        np.testing.assert_array_almost_equal(y_valid, y_test)

        # since d2 provides perfect predictions
        # it should get a higher weight
        # so that y_valid should be exactly y_valid_d2
        y_valid_d2 = ensbuilder.read_preds[d2][Y_VALID][:, 1]
        np.testing.assert_array_almost_equal(y_valid, y_valid_d2)

    def testMain(self):

        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=3,  # Multilabel Classification
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=2,
            max_models_on_disc=None,
            )
        ensbuilder.SAVE2DISC = False

        run_history, ensemble_nbest = ensbuilder.main(time_left=np.inf, iteration=1)

        self.assertEqual(len(ensbuilder.read_preds), 3)
        self.assertIsNotNone(ensbuilder.last_hash)
        self.assertIsNotNone(ensbuilder.y_true_ensemble)

        # Make sure the run history is ok

        # We expect at least 1 element to be in the ensemble
        self.assertGreater(len(run_history), 0)

        # As the data loader loads the same val/train/test
        # we expect 1.0 as score and all keys available
        expected_performance = {
            'ensemble_val_score': 1.0,
            'ensemble_test_score': 1.0,
            'ensemble_optimization_score': 1.0,
        }
        self.assertDictContainsSubset(expected_performance, run_history[0])
        self.assertIn('Timestamp', run_history[0])
        self.assertIsInstance(run_history[0]['Timestamp'], pd.Timestamp)

    def testLimit(self):
        ensbuilder = EnsembleBuilderMemMock(backend=self.backend,
                                            dataset_name="TEST",
                                            task_type=1,  # Binary Classification
                                            metric=roc_auc,
                                            seed=0,  # important to find the test files
                                            ensemble_nbest=10,
                                            # small to trigger MemoryException
                                            memory_limit=10
                                            )
        ensbuilder.SAVE2DISC = False

        ensbuilder.run(time_left=1000, iteration=0)

        # it should try to reduce ensemble_nbest until it also failed at 2
        self.assertEqual(ensbuilder.ensemble_nbest, 1)


class EnsembleSelectionTest(unittest.TestCase):
    def testPredict(self):
        # Test that ensemble prediction applies weights correctly to given
        # predictions. There are two possible cases:
        # 1) predictions.shape[0] == len(self.weights_). In this case,
        # predictions include those made by zero-weighted models. Therefore,
        # we simply apply each weights to the corresponding model preds.
        # 2) predictions.shape[0] < len(self.weights_). In this case,
        # predictions exclude those made by zero-weighted models. Therefore,
        # we first exclude all occurrences of zero in self.weights_, and then
        # apply the weights.
        # If none of the above is the case, predict() raises Error.
        ensemble = EnsembleSelection(ensemble_size=3,
                                     task_type=1,
                                     random_state=np.random.RandomState(0),
                                     metric=accuracy,
                                     )
        # Test for case 1. Create (3, 2, 2) predictions.
        per_model_pred = np.array([
            [[0.9, 0.1],
             [0.4, 0.6]],
            [[0.8, 0.2],
             [0.3, 0.7]],
            [[1.0, 0.0],
             [0.1, 0.9]]
        ])
        # Weights of 3 hypothetical models
        ensemble.weights_ = [0.7, 0.2, 0.1]
        pred = ensemble.predict(per_model_pred)
        truth = np.array([[0.89, 0.11],  # This should be the true prediction.
                          [0.35, 0.65]])
        self.assertTrue(np.allclose(pred, truth))

        # Test for case 2.
        per_model_pred = np.array([
            [[0.9, 0.1],
             [0.4, 0.6]],
            [[0.8, 0.2],
             [0.3, 0.7]],
            [[1.0, 0.0],
             [0.1, 0.9]]
        ])
        # The third model now has weight of zero.
        ensemble.weights_ = [0.7, 0.2, 0.0, 0.1]
        pred = ensemble.predict(per_model_pred)
        truth = np.array([[0.89, 0.11],
                          [0.35, 0.65]])
        self.assertTrue(np.allclose(pred, truth))

        # Test for error case.
        per_model_pred = np.array([
            [[0.9, 0.1],
             [0.4, 0.6]],
            [[0.8, 0.2],
             [0.3, 0.7]],
            [[1.0, 0.0],
             [0.1, 0.9]]
        ])
        # Now the weights have 2 zero weights and 2 non-zero weights,
        # which is incompatible.
        ensemble.weights_ = [0.6, 0.0, 0.0, 0.4]

        with self.assertRaises(ValueError):
            ensemble.predict(per_model_pred)


class SingleBestTest(unittest.TestCase):
    def setUp(self):
        self.run_history = RunHistory()
        self.run_history._add(
            RunKey(
                config_id=3,
                instance_id='{"task_id": "breast_cancer"}',
                seed=1,
                budget=3.0
            ),
            RunValue(
                cost=0.11347517730496459,
                time=0.21858787536621094,
                status=None,
                starttime=time.time(),
                endtime=time.time(),
                additional_info={
                    'duration': 0.20323538780212402,
                    'num_run': 3,
                    'configuration_origin': 'Random Search'}
            ),
            status=None,
            origin=None,
        )
        self.run_history._add(
            RunKey(
                config_id=6,
                instance_id='{"task_id": "breast_cancer"}',
                seed=1,
                budget=6.0
            ),
            RunValue(
                cost=2*0.11347517730496459,
                time=2*0.21858787536621094,
                status=None,
                starttime=time.time(),
                endtime=time.time(),
                additional_info={
                    'duration': 0.20323538780212402,
                    'num_run': 6,
                    'configuration_origin': 'Random Search'}
            ),
            status=None,
            origin=None,
        )

    @unittest.mock.patch('os.path.exists')
    def test_get_identifiers_from_run_history_accuracy(self, exists):
        exists.return_value = True
        ensemble = SingleBest(
             metric=accuracy,
             random_state=1,
             run_history=self.run_history,
             model_dir='/tmp',
        )

        # Just one model
        self.assertEqual(len(ensemble.identifiers_), 1)

        # That model must be the best
        seed, num_run, budget = ensemble.identifiers_[0]
        self.assertEqual(num_run, 3)
        self.assertEqual(seed, 1)
        self.assertEqual(budget, 3.0)

    @unittest.mock.patch('os.path.exists')
    def test_get_identifiers_from_run_history_log_loss(self, exists):
        exists.return_value = True
        ensemble = SingleBest(
             metric=log_loss,
             random_state=1,
             run_history=self.run_history,
             model_dir='/tmp',
        )

        # Just one model
        self.assertEqual(len(ensemble.identifiers_), 1)

        # That model must be the best
        seed, num_run, budget = ensemble.identifiers_[0]
        self.assertEqual(num_run, 3)
        self.assertEqual(seed, 1)
        self.assertEqual(budget, 3.0)


class EnsembleProcessBuilderTest(unittest.TestCase):
    def setUp(self):
        self.backend = BackendMock()

    def tearDown(self):
        pass

    def test_ensemble_builder_process_termination_request(self):
        """
        Makes sure we can kill an ensemble process via a event
        """
        client = dask.distributed.Client(n_workers=1, processes=True)
        event = dask.distributed.Event('None')

        # Set the event so the run does not even start
        event.set()

        ensemble = ensemble_builder_process(
            start_time=time.time(),
            time_left_for_ensembles=1000,
            sleep_duration=2,
            event='None',
            backend=self.backend,
            dataset_name='Test',
            task=1,
            metric=roc_auc,
            ensemble_size=50,
            ensemble_nbest=10,
            max_models_on_disc=None,
            seed=0,
            precision=32,
            max_iterations=1,
            read_at_most=np.inf,
            ensemble_memory_limit=10,
            random_state=0,
            logger_name='Ensemblebuilder',
        )

        # make sure message is in log file
        msg = 'Terminating ensemble building as SMAC process is done'
        logger_name = 'autosklearn.ensemble_builder'
        logfile = os.path.join(
            self.backend.temporary_directory,
            '%s.log' % str(logger_name)
        )
        with open(logfile) as f:
            self.assertIn(msg,  f.read())

        # Also makes sure the ensemble does not return any history
        self.assertEqual(ensemble, [])
        client.close()

    def test_ensemble_builder_process_realrun(self):
        dask.config.set({'distributed.worker.daemon': False})
        client = dask.distributed.Client(n_workers=2, processes=True)
        ensemble = client.submit(
            ensemble_builder_process,
            start_time=time.time(),
            time_left_for_ensembles=1000,
            sleep_duration=2,
            event='None',
            backend=self.backend,
            dataset_name='Test',
            task=1,
            metric=MockMetric,
            ensemble_size=50,
            ensemble_nbest=10,
            max_models_on_disc=None,
            seed=0,
            precision=32,
            max_iterations=1,
            read_at_most=np.inf,
            ensemble_memory_limit=None,
            random_state=0,
            logger_name='Ensemblebuilder',
        )
        history = ensemble.result()

        self.assertIn('ensemble_optimization_score', history[0])
        self.assertEqual(history[0]['ensemble_optimization_score'], 0.9)
        self.assertIn('ensemble_val_score', history[0])
        self.assertEqual(history[0]['ensemble_val_score'], 0.9)
        self.assertIn('ensemble_test_score', history[0])
        self.assertEqual(history[0]['ensemble_test_score'], 0.9)
        client.close()


if __name__ == '__main__':
    unittest.main()
