import unittest

import numpy as np

from AutoML2015.wrapper import openml_wrapper


class OpenMLWrapperTest(unittest.TestCase):
    def test_load_dataset(self):
        X, y, categorical = openml_wrapper.load_dataset("2")
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(categorical, list)
        self.assertEqual(len(categorical), X.shape[1])

    def test_create_datamanager(self):
        # Just make sure it works
        X, y, categorical = openml_wrapper.load_dataset("2")
        D = openml_wrapper.create_mock_data_manager(X, y, categorical,
            "bac_metric", "binary.classification")

    def test_remove_categorical(self):
        X, y, categorical = openml_wrapper.load_dataset("2")
        X, categorical = openml_wrapper.remove_categorical_features(X, categorical)
        self.assertEqual((898, 6), X.shape)
        self.assertEqual(len(categorical), 6)
        self.assertListEqual([False]*6, categorical)

    def test_main(self):
        args = {'dataset': '2', 'metric': 'bac_metric', 'task_type':
            'multiclass.classification', 'fold': 0, 'folds': 5,
            'remove_categorical': None}
        params = {'preprocessor': 'None', 'classifier': 'random_forest',
                  'rescaling:strategy': 'standard', 'imputation:strategy':
                  'mean', 'random_forest:criterion': 'gini',
                  'random_forest:max_features': '1',
                  'random_forest:min_samples_leaf': '1',
                  'random_forest:min_samples_split': '2',
                  'random_forest:bootstrap': 'True',
                  'random_forest:max_depth': 'None',
                  'random_forest:n_estimators': '100',
                  'random_forest:max_leaf_nodes': 'None'}
        err, additional_run_info = openml_wrapper.main(args, params)
        print err
        print additional_run_info
