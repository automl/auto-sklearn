# -*- encoding: utf-8 -*-
import os
import sys
import unittest
import unittest.mock

import numpy as np


this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_multiclass_classification_datamanager, \
    get_regression_datamanager
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.metrics import accuracy, r2, mean_squared_error


class AbstractEvaluatorTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_finish_up_model_predicts_NaN(self):
        '''Tests by handing in predictions which contain NaNs'''
        rs = np.random.RandomState(1)
        D = get_multiclass_classification_datamanager()

        backend_api = unittest.mock.Mock()
        queue_mock = unittest.mock.Mock()
        ae = AbstractEvaluator(datamanager=D, backend=backend_api,
                               output_y_hat_optimization=False,
                               queue=queue_mock, metric=accuracy)
        ae.Y_optimization = rs.rand(33, 3)
        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        # NaNs in prediction ensemble
        predictions_ensemble[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            0.1, predictions_ensemble, predictions_valid, predictions_test)
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for optimization set '
                                   'contains NaNs.'})

        # NaNs in prediction validation
        predictions_ensemble[5, 2] = 0.5
        predictions_valid[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            0.1, predictions_ensemble, predictions_valid, predictions_test)
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for validation set '
                                   'contains NaNs.'})

        # NaNs in prediction test
        predictions_valid[5, 2] = 0.5
        predictions_test[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            0.1, predictions_ensemble, predictions_valid, predictions_test)
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for test set contains '
                                   'NaNs.'})

        self.assertEqual(backend_api.save_predictions_as_npy.call_count, 0)

    @unittest.mock.patch('os.path.exists')
    def test_disable_file_output(self, exists_mock):
        backend_mock = unittest.mock.Mock()
        backend_mock.get_model_dir.return_value = 'abc'
        queue_mock = unittest.mock.Mock()

        rs = np.random.RandomState(1)
        D = get_multiclass_classification_datamanager()

        ae = AbstractEvaluator(datamanager=D, backend=backend_mock,
                               queue=queue_mock, disable_file_output=True,
                               metric=accuracy)

        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        loss_, additional_run_info_ = ae.file_output(
            predictions_ensemble, predictions_valid, predictions_test)

        self.assertIsNone(loss_)
        self.assertEqual(additional_run_info_, {})
        # This function is not guarded by a an if statement
        self.assertEqual(backend_mock.save_predictions_as_npy.call_count, 0)
        self.assertEqual(backend_mock.save_model.call_count, 0)

        ae = AbstractEvaluator(datamanager=D, backend=backend_mock,
                               output_y_hat_optimization=False,
                               queue=queue_mock, disable_file_output=['model'],
                               metric=accuracy)
        ae.Y_optimization = predictions_ensemble

        loss_, additional_run_info_ = ae.file_output(
            predictions_ensemble, predictions_valid, predictions_test)

        self.assertIsNone(loss_)
        self.assertIsNone(additional_run_info_)
        # This function is not guarded by a an if statement
        self.assertEqual(backend_mock.save_predictions_as_npy.call_count, 3)
        self.assertEqual(backend_mock.save_model.call_count, 0)

        ae = AbstractEvaluator(datamanager=D, backend=backend_mock,
                               output_y_hat_optimization=False,
                               queue=queue_mock, metric=accuracy,
                               disable_file_output=['y_optimization'])
        exists_mock.return_value = True
        ae.Y_optimization = predictions_ensemble
        ae.model = 'model'

        loss_, additional_run_info_ = ae.file_output(
            predictions_ensemble, predictions_valid, predictions_test)

        self.assertIsNone(loss_)
        self.assertIsNone(additional_run_info_)
        # This function is not guarded by a an if statement
        self.assertEqual(backend_mock.save_predictions_as_npy.call_count, 5)
        self.assertEqual(backend_mock.save_model.call_count, 1)
