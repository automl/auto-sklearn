from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UnParametrizedHyperparameter

from autosklearn.pipeline.components.algorithms import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import SelectPercentileBase
from autosklearn.pipeline.constants import *


class SelectPercentileRegression(SelectPercentileBase,
                                 AutoSklearnPreprocessingAlgorithm):

    def __init__(self):
        """ Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """
        import sklearn.feature_selection
        super(SelectPercentileRegression, self).__init__()
        self.random_state = None  # We don't use this
        self.percentile = None
        self.score_func = None

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SPR',
                'name': 'Select Percentile Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SIGNED_DATA),
                'output': (INPUT,)}

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
