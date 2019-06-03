from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SIGNED_DATA, UNSIGNED_DATA, SPARSE, DENSE, INPUT


class SelectRates(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, alpha, mode='fpr',
                 score_func="chi2", task="classification", random_state=None):
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.task = task

        if score_func == "chi2" and task == "classification":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif" and task == "classification":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info_classif" and task == "classification":
            self.score_func = sklearn.feature_selection.mutual_info_classif
        elif score_func == "f_regression" and task == "regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info_regression" and task == "regression":
            self.score_func = sklearn.feature_selection.mutual_info_regression
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info_classif') "
                             "for task='classification', "
                             "or in ('f_regression, 'mutual_info_regression') "
                             "for task='regression', "
                             "but is: %s for task='%s'" % (score_func, task))

        self.mode = mode

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
        import scipy.sparse
        import sklearn.feature_selection

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

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

        if dataset_properties is not None and \
            'target_type' in dataset_properties and \
                dataset_properties['target_type'] == 'regression':

            task = 'regression'
        else:
            task = 'classification'

        return {'shortname': 'SR',
                'name': 'Univariate Feature Selection based on rates',
                'handles_regression': (task == 'regression'),
                'handles_classification': (task == 'classification'),
                'handles_multiclass': (task == 'classification'),
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, data_type),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1)

        if dataset_properties is not None and \
            'target_type' in dataset_properties and \
                dataset_properties['target_type'] == 'regression':

            score_func = Constant(
                    name="score_func", value="f_regression")
        else:
            score_func = CategoricalHyperparameter(
                name="score_func",
                choices=["chi2", "f_classif"],
                default_value="chi2")

        if dataset_properties is not None:
            # Chi2 can handle sparse data, so we respect this
            if 'sparse' in dataset_properties and \
                    dataset_properties['sparse'] and \
                ('target_type' not in dataset_properties or \
                    dataset_properties['target_type'] == 'classification'):

                score_func = Constant(
                    name="score_func", value="chi2")

        mode = CategoricalHyperparameter('mode', ['fpr', 'fdr', 'fwe'], 'fpr')

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        return cs
