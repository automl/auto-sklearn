from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SIGNED_DATA, UNSIGNED_DATA, SPARSE, DENSE, INPUT


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
            # Work Around as SMAC does not handle Not Equal
            # Mutual info needs scikit learn default to prevent
            # running into p_values problem (no pvalue found)
            self.mode = 'percentile'
        else:
            raise ValueError("score_func must be in ('f_regression, 'mutual_info_regression') "
                             "for task=regression "
                             "but is: %s " % (score_func))

    def fit(self, X, y):
        import scipy.sparse
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

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
        data_type = UNSIGNED_DATA

        if dataset_properties is not None:
            signed = dataset_properties.get('signed')
            if signed is not None:
                data_type = SIGNED_DATA if signed is True else UNSIGNED_DATA

        return {'shortname': 'SR',
                'name': 'Univariate Feature Selection based on rates',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, data_type),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1)

        if dataset_properties is not None and 'sparse' in dataset_properties \
                and dataset_properties['sparse']:
            choices = ['mutual_info_regression']
        else:
            choices = ['f_regression', 'mutual_info_regression']

        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=choices,
            default_value="f_regression" if 'f_regression' in choices else choices[0])

        mode = CategoricalHyperparameter('mode', ['fpr', 'fdr', 'fwe'], 'fpr')

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # In case of mutual info regression, the mode needs to be percentile
        # Which is the scikit learn default, else we run into p_values problem
        # SMAC Cannot handle OR, so leave this code here for the future.
        # Right now, we will have mode in the config space when we
        # have mutual_info, yet it is not needed
        # if 'mutual_info_regression' in choices:
        #     cond = NotEqualsCondition(mode, score_func, 'mutual_info_regression')
        #     cs.add_condition(cond)
        # if 'mutual_info_classif' in choices:
        #     cond = NotEqualsCondition(mode, score_func, 'mutual_info_classif')
        #     cs.add_condition(cond)

        return cs
