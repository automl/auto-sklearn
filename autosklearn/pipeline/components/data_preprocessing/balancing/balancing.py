import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class Balancing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, strategy='none', random_state=None):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_weights(self, Y, classifier, preprocessor, init_params, fit_params):
        if init_params is None:
            init_params = {}

        if fit_params is None:
            fit_params = {}

        # Classifiers which require sample weights:
        # We can have adaboost in here, because in the fit method,
        # the sample weights are normalized:
        # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/ensemble/weight_boosting.py#L121
        # Have RF and ET in here because they emit a warning if class_weights
        #  are used together with warmstarts
        clf_ = ['adaboost', 'gradient_boosting', 'random_forest',
                'extra_trees', 'sgd', 'passive_aggressive', 'xgradient_boosting']
        pre_ = []
        if classifier in clf_ or preprocessor in pre_:
            if len(Y.shape) > 1:
                offsets = [2 ** i for i in range(Y.shape[1])]
                Y_ = np.sum(Y * offsets, axis=1)
            else:
                Y_ = Y

            unique, counts = np.unique(Y_, return_counts=True)
            # This will result in an average weight of 1!
            cw = 1 / (counts / np.sum(counts)) / 2
            if len(Y.shape) == 2:
                cw /= Y.shape[1]

            sample_weights = np.ones(Y_.shape)

            for i, ue in enumerate(unique):
                mask = Y_ == ue
                sample_weights[mask] *= cw[i]

            if classifier in clf_:
                fit_params['classifier:sample_weight'] = sample_weights
            if preprocessor in pre_:
                fit_params['preprocessor:sample_weight'] = sample_weights

        # Classifiers which can adjust sample weights themselves via the
        # argument `class_weight`
        clf_ = ['decision_tree', 'liblinear_svc',
                'libsvm_svc']
        pre_ = ['liblinear_svc_preprocessor',
                'extra_trees_preproc_for_classification']
        if classifier in clf_:
            init_params['classifier:class_weight'] = 'balanced'
        if preprocessor in pre_:
            init_params['preprocessor:class_weight'] = 'balanced'

        clf_ = ['ridge']
        if classifier in clf_:
            class_weights = {}

            unique, counts = np.unique(Y, return_counts=True)
            cw = 1. / counts
            cw = cw / np.mean(cw)

            for i, ue in enumerate(unique):
                class_weights[ue] = cw[i]

            if classifier in clf_:
                init_params['classifier:class_weight'] = class_weights

        return init_params, fit_params

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Balancing',
                'name': 'Balancing Imbalanced Class Distributions',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["none", "weighting"], default_value="none")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs

