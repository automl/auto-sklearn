from __future__ import print_function

import copy
import os
import shutil
import sys
import unittest

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

import autosklearn.cli.base_interface

try:
    import __builtin__
    _ = __builtin__.print
    patch_string = '__builtin__.print'
except ImportError:
    import builtins
    _ = builtins.print
    patch_string = 'builtins.print'

class Base_interfaceTest(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../.data')
        self.dataset = '31_bac'
        self.dataset_string = os.path.join(self.data_dir, self.dataset)

        self.params = {
            'balancing:strategy': 'none',
            'classifier:__choice__': 'random_forest',
            'imputation:strategy': 'mean',
            'preprocessor:__choice__': 'no_preprocessing',
            'classifier:random_forest:bootstrap': 'True',
            'classifier:random_forest:criterion': 'gini',
            'classifier:random_forest:max_depth': 'None',
            'classifier:random_forest:max_features': '1.0',
            'classifier:random_forest:max_leaf_nodes': 'None',
            'classifier:random_forest:min_weight_fraction_leaf': '0.0',
            'classifier:random_forest:min_samples_leaf': '1',
            'classifier:random_forest:min_samples_split': '2',
            'classifier:random_forest:n_estimators': '100',
            'one_hot_encoding:use_minimum_fraction': 'True',
            'one_hot_encoding:minimum_fraction': '0.01',
            'rescaling:__choice__': 'min/max'
        }
        self.output_directory = os.path.join(os.getcwd(),
                                             '.test_base_interface')

        try:
            shutil.rmtree(self.output_directory)
        except Exception:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.output_directory)
        except Exception:
            pass

    @mock.patch(patch_string)
    def test_holdout(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'holdout',
                                            '1',
                                            self.params,
                                            output_dir=self.output_directory)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.755128', result)

    @mock.patch(patch_string)
    def test_holdout_iterative_fit(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'holdout-iterative-fit',
                                            '1',
                                            self.params,
                                            output_dir=self.output_directory)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.725277', result)

    @mock.patch(patch_string)
    def test_testset(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'test',
                                            '1',
                                            self.params,
                                            output_dir=self.output_directory)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.772006', result)

    @mock.patch(patch_string)
    def test_cv(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'cv',
                                            '1',
                                            self.params,
                                            mode_args={'folds': 3},
                                            output_dir=self.output_directory)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.766880', result)

    @mock.patch(patch_string)
    def test_partial_cv(self, patch):
        results = []
        for fold in range(3):
            params = copy.deepcopy(self.params)
            autosklearn.cli.base_interface.main(self.dataset_string,
                                                'partial-cv',
                                                '1',
                                                params,
                                                mode_args={'folds': 3,
                                                           'fold': fold},
                                                output_dir=self.output_directory)
            # Returns the actual call
            call_args = patch.call_args[0][0]
            result = call_args.split(",")[3].strip()
            results.append(result)

        self.assertEqual(['0.780112', '0.791236', '0.729430'], results)

    @mock.patch(patch_string)
    def test_nested_cv(self, patch):
        autosklearn.cli.base_interface.main(self.dataset_string,
                                            'nested-cv',
                                            '1',
                                            self.params,
                                            mode_args={'outer_folds': 3,
                                                       'inner_folds': 3},
                                            output_dir=self.output_directory)
        # Returns the actual call
        call_args = patch.call_args[0][0]
        result = call_args.split(",")[3].strip()
        self.assertEqual('0.811493', result)

