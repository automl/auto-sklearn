from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import \
    SelectPercentileBase
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT


class SelectPercentileRegression(SelectPercentileBase,
                                 AutoSklearnPreprocessingAlgorithm):

    def __init__(self, percentile, score_func="f_regression", random_state=None):
        """ Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.percentile = int(float(percentile))
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_regression
        else:
            raise ValueError("Don't know this scoring function: %s" % score_func)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SPR',
                'name': 'Select Percentile Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        percentile = UniformFloatHyperparameter(
            "percentile", lower=1, upper=99, default_value=50)

        score_func = CategoricalHyperparameter(
            name="score_func", choices=["f_regression", "mutual_info"])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([percentile, score_func])
        return cs
