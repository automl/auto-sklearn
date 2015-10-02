# -*- encoding: utf-8 -*-
from __future__ import print_function

import unittest

from ParamSklearn.util import get_dataset

from autosklearn.constants import *
from autosklearn.metalearning.mismbo import calc_meta_features, \
    calc_meta_features_encoded, \
    create_metalearning_string_for_smac_call
from autosklearn.util.paramsklearn import get_configuration_space


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

    def test_metalearning(self):
        dataset_name = 'digits'

        initial_challengers = {
            'acc_metric': ["--initial-challengers \" "
                           "-adaboost:algorithm 'SAMME.R' "
                           "-adaboost:learning_rate '0.400363929326' "
                           "-adaboost:max_depth '5' "
                           "-adaboost:n_estimators '319' "
                           "-balancing:strategy 'none' "
                           "-classifier 'adaboost' "
                           "-imputation:strategy 'most_frequent' "
                           "-preprocessor 'no_preprocessing' "
                           "-rescaling:strategy 'min/max'\""],
            'auc_metric': ["--initial-challengers \" "
                           "-adaboost:algorithm 'SAMME.R' "
                           "-adaboost:learning_rate '0.966883114819' "
                           "-adaboost:max_depth '5' "
                           "-adaboost:n_estimators '412' "
                           "-balancing:strategy 'weighting' "
                           "-classifier 'adaboost' "
                           "-imputation:strategy 'median' "
                           "-preprocessor 'no_preprocessing' "
                           "-rescaling:strategy 'min/max'\""],
            'bac_metric': ["--initial-challengers \" "
                           "-adaboost:algorithm 'SAMME.R' "
                           "-adaboost:learning_rate '0.400363929326' "
                           "-adaboost:max_depth '5' "
                           "-adaboost:n_estimators '319' "
                           "-balancing:strategy 'none' "
                           "-classifier 'adaboost' "
                           "-imputation:strategy 'most_frequent' "
                           "-preprocessor 'no_preprocessing' "
                           "-rescaling:strategy 'min/max'\""],
            'f1_metric': ["--initial-challengers \" "
                          "-adaboost:algorithm 'SAMME.R' "
                          "-adaboost:learning_rate '0.966883114819' "
                          "-adaboost:max_depth '5' "
                          "-adaboost:n_estimators '412' "
                          "-balancing:strategy 'weighting' "
                          "-classifier 'adaboost' "
                          "-imputation:strategy 'median' "
                          "-preprocessor 'no_preprocessing' "
                          "-rescaling:strategy 'min/max'\""],
            'pac_metric': ["--initial-challengers \" "
                           "-adaboost:algorithm 'SAMME.R' "
                           "-adaboost:learning_rate '0.400363929326' "
                           "-adaboost:max_depth '5' "
                           "-adaboost:n_estimators '319' "
                           "-balancing:strategy 'none' "
                           "-classifier 'adaboost' "
                           "-imputation:strategy 'most_frequent' "
                           "-preprocessor 'no_preprocessing' "
                           "-rescaling:strategy 'min/max'\""]
        }

        for metric in initial_challengers:
            configuration_space = get_configuration_space(
                {
                    'metric': metric,
                    'task': MULTICLASS_CLASSIFICATION,
                    'is_sparse': False
                },
                include_preprocessors=['no_preprocessing'])

            X_train, Y_train, X_test, Y_test = get_dataset(dataset_name)
            categorical = [False] * X_train.shape[1]

            meta_features_label = calc_meta_features(X_train, Y_train,
                                                     categorical, dataset_name)
            meta_features_encoded_label = calc_meta_features_encoded(X_train,
                                                                     Y_train,
                                                                     categorical,
                                                                     dataset_name)
            initial_configuration_strings_for_smac = \
                create_metalearning_string_for_smac_call(
                    meta_features_label,
                    meta_features_encoded_label,
                    configuration_space, dataset_name, metric,
                    MULTICLASS_CLASSIFICATION, False, 1, None)

            self.assertEqual(initial_challengers[metric],
                             initial_configuration_strings_for_smac)
