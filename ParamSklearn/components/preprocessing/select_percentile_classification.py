from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, Constant

import sklearn.feature_selection

from ParamSklearn.components.preprocessor_base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.components.preprocessing.select_percentile import SelectPercentileBase
from ParamSklearn.util import DENSE, SPARSE, INPUT


class SelectPercentileClassification(SelectPercentileBase,
                                     ParamSklearnPreprocessingAlgorithm):

    def __init__(self, percentile, score_func="chi2", random_state=None):
        """ Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """
        self.random_state = random_state  # We don't use this
        self.percentile = int(float(percentile))
        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif'), "
                             "but is: %s" % score_func)

    @staticmethod
    def get_properties():
        return {'shortname': 'SPC',
                'name': 'Select Percentile Classification',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE),
                'output': INPUT,
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        percentile = UniformFloatHyperparameter(
            name="percentile", lower=10, upper=90, default=50)

        score_func = CategoricalHyperparameter(
            name="score_func", choices=["chi2", "f_classif"], default="chi2")
        if dataset_properties is not None:
            # Chi2 can handle sparse data, so we respect this
            if 'sparse' in dataset_properties and dataset_properties['sparse']:
                score_func = Constant(
                    name="score_func", value="chi2")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(percentile)
        cs.add_hyperparameter(score_func)

        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name

