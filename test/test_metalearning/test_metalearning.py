# -*- encoding: utf-8 -*-

import unittest

from autosklearn.pipeline.util import get_dataset

from autosklearn.constants import *
from autosklearn.metalearning.mismbo import \
    suggest_via_metalearning
from autosklearn.util.pipeline import get_configuration_space


class MetafeatureValueDummy(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value


class Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = \
            get_dataset('iris')

        eliminate_class_two = self.Y_train != 2
        self.X_train = self.X_train[eliminate_class_two]
        self.Y_train = self.Y_train[eliminate_class_two]

    @unittest.skip('TODO refactor!')
    def test_metalearning(self):
        dataset_name_classification = 'digits'
        initial_challengers_classification = {
            ACC_METRIC: "--initial-challengers \" "
                        "-balancing:strategy 'weighting' "
                        "-classifier:__choice__ 'proj_logit'",
            AUC_METRIC: "--initial-challengers \" "
                        "-balancing:strategy 'weighting' "
                        "-classifier:__choice__ 'liblinear_svc'",
            BAC_METRIC: "--initial-challengers \" "
                        "-balancing:strategy 'weighting' "
                        "-classifier:__choice__ 'proj_logit'",
            F1_METRIC: "--initial-challengers \" "
                       "-balancing:strategy 'weighting' "
                       "-classifier:__choice__ 'proj_logit'",
            PAC_METRIC: "--initial-challengers \" "
                        "-balancing:strategy 'none' "
                        "-classifier:__choice__ 'random_forest'"
        }

        dataset_name_regression = 'diabetes'
        initial_challengers_regression = {
            A_METRIC: "--initial-challengers \" "
                      "-imputation:strategy 'mean' "
                      "-one_hot_encoding:minimum_fraction '0.01' "
                      "-one_hot_encoding:use_minimum_fraction 'True' "
                      "-preprocessor:__choice__ 'no_preprocessing' "
                      "-regressor:__choice__ 'random_forest'",
            R2_METRIC: "--initial-challengers \" "
                       "-imputation:strategy 'mean' "
                       "-one_hot_encoding:minimum_fraction '0.01' "
                       "-one_hot_encoding:use_minimum_fraction 'True' "
                       "-preprocessor:__choice__ 'no_preprocessing' "
                       "-regressor:__choice__ 'random_forest'",
        }

        for dataset_name, task, initial_challengers in [
            (dataset_name_regression, REGRESSION,
             initial_challengers_regression),
            (dataset_name_classification, MULTICLASS_CLASSIFICATION,
             initial_challengers_classification)
            ]:
            for metric in initial_challengers:
                configuration_space = get_configuration_space(
                    {
                        'metric': metric,
                        'task': task,
                        'is_sparse': False
                    },
                    include_preprocessors=['no_preprocessing'])

                X_train, Y_train, X_test, Y_test = get_dataset(dataset_name)
                categorical = [False] * X_train.shape[1]

                meta_features_label = calc_meta_features(
                    X_train, Y_train, categorical, dataset_name, task)
                meta_features_encoded_label = calc_meta_features_encoded(
                    X_train, Y_train, categorical, dataset_name, task)

                initial_configuration_strings_for_smac = \
                    suggest_via_metalearning(
                        meta_features_label,
                        meta_features_encoded_label,
                        configuration_space, dataset_name, metric,
                        task, False, 1, None)

                print(metric)
                print(initial_configuration_strings_for_smac[0])
                self.assertTrue(initial_configuration_strings_for_smac[
                                    0].startswith(initial_challengers[metric]))
