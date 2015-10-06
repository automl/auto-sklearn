from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, UnParametrizedHyperparameter

import sklearn.feature_selection

from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.components.feature_preprocessing.select_percentile import SelectPercentileBase
from ParamSklearn.constants import *


class SelectPercentileRegression(SelectPercentileBase,
                                 ParamSklearnPreprocessingAlgorithm):

    def __init__(self, percentile, score_func="f_classif", random_state=None):
        """ Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """

        self.random_state = random_state  # We don't use this
        self.percentile = int(float(percentile))
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        else:
            raise ValueError("Don't know this scoring function: %s" % score_func)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SPR',
                'name': 'Select Percentile Regression',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        percentile = UniformFloatHyperparameter(
            "percentile", lower=1, upper=99, default=50)

        score_func = UnParametrizedHyperparameter(
            name="score_func", value="f_regression")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(percentile)
        cs.add_hyperparameter(score_func)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name

