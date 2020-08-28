from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace import NotEqualsCondition

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import UNSIGNED_DATA, SPARSE, DENSE, INPUT


class SelectRegressionRates(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, alpha, mode='percentile',
                 score_func="f_regression", random_state=None):
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.mode = mode

        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info_regression":
            self.score_func = sklearn.feature_selection.mutual_info_regression
            # Mutual info consistently crashes if percentile is not the mode
            self.mode = 'percentile'
        else:
            raise ValueError("score_func must be in ('f_regression, 'mutual_info_regression') "
                             "for task=regression "
                             "but is: %s " % (score_func))

    def fit(self, X, y):
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode)

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()
        try:
            Xt = self.preprocessor.transform(X)
        except ValueError as e:
            if "zero-size array to reduction operation maximum which has no " \
                    "identity" in e.message:
                raise ValueError(
                    "%s removed all features." % self.__class__.__name__)
            else:
                raise e

        if Xt.shape[1] == 0:
            raise ValueError(
                "%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SR',
                'name': 'Univariate Feature Selection based on rates',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1)

        if dataset_properties is not None and dataset_properties.get('sparse'):
            choices = ['mutual_info_regression', 'f_regression']
        else:
            choices = ['f_regression']

        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=choices,
            default_value="f_regression")

        mode = CategoricalHyperparameter('mode', ['fpr', 'fdr', 'fwe'], 'fpr')

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # Mutual info consistently crashes if percentile is not the mode
        if 'mutual_info_regression' in choices:
            cond = NotEqualsCondition(mode, score_func, 'mutual_info_regression')
            cs.add_condition(cond)

        return cs
