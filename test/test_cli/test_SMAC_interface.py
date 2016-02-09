# -*- encoding: utf-8 -*-
from __future__ import print_function

import os
import unittest
import shlex
import shutil
import sys

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

import autosklearn.cli.SMAC_interface as SMAC_interface


class SMAC_interfaceTest(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../.data')
        self.dataset = '31_bac'
        self.dataset_string = os.path.join(self.data_dir, self.dataset)
        self.param_string = '-balancing:strategy none ' \
                            '-classifier random_forest ' \
                            '-imputation:strategy mean ' \
                            '-preprocessor no_preprocessing ' \
                            '-random_forest:bootstrap True ' \
                            '-random_forest:criterion gini ' \
                            '-random_forest:max_depth None ' \
                            '-random_forest:max_features 1.0 ' \
                            '-random_forest:max_leaf_nodes None ' \
                            '-random_forest:min_samples_leaf 1 ' \
                            '-random_forest:min_samples_split 2 ' \
                            '-random_forest:n_estimators 100 ' \
                            '-rescaling:strategy min/max'
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
        self.output_directory = os.path.join(os.getcwd(),
                                             '.test_SMAC_interface')

        try:
            shutil.rmtree(self.output_directory)
        except Exception:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.output_directory)
        except Exception:
            pass

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_holdout(self, patch):
        call = 'autosklearn.cli.SMAC_interface holdout %s ' \
               '1000000 0 -1 %s' % \
               (self.dataset_string, self.param_string)
        sys.argv = shlex.split(call)

        SMAC_interface.main(output_dir=self.output_directory)
        self.assertEqual(patch.call_count, 1)
        call_args, call_kwargs = patch.call_args
        self.assertEqual(call_args, (self.dataset_string, 'holdout', 1,
                                     self.params))
        self.assertEqual(call_kwargs, {'mode_args': None,
                                       'output_dir': self.output_directory})

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_holdout_iterative_fit(self, patch):
        call = 'autosklearn.cli.SMAC_interface holdout-iterative-fit %s ' \
               '1000000 0 -1 %s' % \
               (self.dataset_string, self.param_string)
        sys.argv = shlex.split(call)

        SMAC_interface.main(output_dir=self.output_directory)
        self.assertEqual(patch.call_count, 1)
        call_args, call_kwargs = patch.call_args
        self.assertEqual(call_args, (self.dataset_string,
                                     'holdout-iterative-fit', 1,
                                     self.params))
        self.assertEqual(call_kwargs, {'mode_args': None,
                                       'output_dir': self.output_directory})

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_testset(self, patch):
        call = 'autosklearn.cli.SMAC_interface test %s ' \
               '1000000 0 -1 %s' % \
               (self.dataset_string, self.param_string)
        sys.argv = shlex.split(call)

        SMAC_interface.main(output_dir=self.output_directory)
        self.assertEqual(patch.call_count, 1)
        call_args, call_kwargs = patch.call_args
        self.assertEqual(call_args, (self.dataset_string, 'test', 1,
                                     self.params))
        self.assertEqual(call_kwargs, {'mode_args': None,
                                       'output_dir': self.output_directory})

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_cv(self, patch):
        call = 'autosklearn.cli.SMAC_interface cv:3 %s ' \
               '1000000 0 -1 %s' % \
               (self.dataset_string, self.param_string)
        sys.argv = shlex.split(call)

        SMAC_interface.main(output_dir=self.output_directory)
        self.assertEqual(patch.call_count, 1)
        call_args, call_kwargs = patch.call_args
        self.assertEqual(call_args, (self.dataset_string, 'cv', 1,
                                     self.params))
        self.assertEqual(call_kwargs, {'mode_args': {'folds': 3},
                                       'output_dir': self.output_directory})

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_partial_cv(self, patch):
        for fold in range(3):
            call = 'autosklearn.cli.SMAC_interface partial-cv:%d/3 %s ' \
                   '1000000 0 -1 %s' % \
                   (fold, self.dataset_string, self.param_string)
            sys.argv = shlex.split(call)

            SMAC_interface.main(output_dir=self.output_directory)
            self.assertEqual(patch.call_count, fold + 1)
            call_args, call_kwargs = patch.call_args
            self.assertEqual(call_args, (self.dataset_string, 'partial-cv', 1,
                                         self.params))
            self.assertEqual(call_kwargs, {'mode_args': {'folds': 3,
                                                         'fold': fold},
                                           'output_dir': self.output_directory})

    @mock.patch('autosklearn.cli.base_interface.main')
    def test_nested_cv(self, patch):
        call = 'autosklearn.cli.SMAC_interface nested-cv:3/3 %s ' \
               '1000000 0 -1 %s' % \
               (self.dataset_string, self.param_string)
        sys.argv = shlex.split(call)

        SMAC_interface.main(output_dir=self.output_directory)
        self.assertEqual(patch.call_count, 1)
        call_args, call_kwargs = patch.call_args
        self.assertEqual(call_args, (self.dataset_string, 'nested-cv', 1,
                                     self.params))
        self.assertEqual(call_kwargs, {'mode_args': {'outer_folds': 3,
                                                     'inner_folds': 3},
                                       'output_dir': self.output_directory})
