from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, Constant

import sklearn.feature_selection

from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.components.feature_preprocessing.select_percentile import SelectPercentileBase
from ParamSklearn.constants import *

import scipy.sparse


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

    def fit(self, X, y):
        self.preprocessor = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func,
                percentile=self.percentile)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data<0] = 0.0
            else:
                X[X<0] = 0.0

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        Xt = self.preprocessor.transform(X)
        if Xt.shape[1] == 0:
            raise ValueError(
                "%s removed all features." % self.__class__.__name__)
        return Xt


    @staticmethod
    def get_properties(dataset_properties=None):
        data_type = UNSIGNED_DATA
        if dataset_properties is not None:
            signed = dataset_properties.get('signed')
            if signed is not None:
                data_type = SIGNED_DATA if signed is True else UNSIGNED_DATA

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
                'input': (SPARSE, DENSE, data_type),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        percentile = UniformFloatHyperparameter(
            name="percentile", lower=1, upper=99, default=50)

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

