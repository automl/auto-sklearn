from __future__ import print_function

import os
import sys
import unittest

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

import autosklearn.cli.base_interface

class Base_interfaceTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../.data')
        self.dataset = '31_bac'
        self.dataset_string = os.path.join(self.data_dir, self.dataset)

        self.params = {
            'balancing:strategy': 'none',
            'classifier': 'random_forest',
            'imputation:strategy': 'mean',
            'preprocessor': 'no_preprocessing',
            'random_forest:bootstrap': 'True',
            'random_forest:criterion': 'gini',
            'random_forest:max_depth': 'None',
            'random_forest:max_features': '1.0',
            'random_forest:max_leaf_nodes': 'None',
            'random_forest:min_samples_leaf': '1',
            'random_forest:min_samples_split': '2',
            'random_forest:n_estimators': '100',
            'rescaling:strategy': 'min/max'
        }

    def tearDown(self):
        try:
            manager = os.path.join(os.path.dirname(__file__),
                                   '%s_Manager.pkl' % self.dataset)
            os.remove(manager)
        except Exception:
            pass

    @mock.patch('__builtin__.print')
    def test_holdout(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'holdout',
                                            '1',
                                            self.params)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.740202', result)

    @mock.patch('__builtin__.print')
    def test_testset(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'test',
                                            '1',
                                            self.params)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.670996', result)

    @mock.patch('__builtin__.print')
    def test_cv(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'cv',
                                            '1',
                                            self.params,
                                            mode_args={'folds': 3})
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.779673', result)

    @mock.patch('__builtin__.print')
    def test_partial_cv(self, patch):
        results = []
        for fold in range(3):
            autosklearn.cli.base_interface.main(self.dataset_string,
                                                'partial-cv',
                                                '1',
                                                self.params,
                                                mode_args={'folds': 3,
                                                           'fold': fold})
            # Returns the actual call
            call_args = patch.call_args[0][0]
            result = call_args.split(",")[3].strip()
            results.append(result)

        self.assertEqual(['0.795038', '0.827497', '0.716609'], results)

    @mock.patch('__builtin__.print')
    def test_nested_cv(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'nested-cv',
                                            '1',
                                            self.params,
                                            mode_args={'outer_folds': 3,
                                                       'inner_folds': 3})
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.815061', result)

