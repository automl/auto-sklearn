import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class SelectKBestComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, features: str = "missing-only",
                 random_state=None):
        super().__init__()
        self.features = features
        self.missing_values = missing_values
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression
        if self.score_func == "chi2":
            score_func = chi2
        elif self.score_func == "f_classif":
            score_func = f_classif
        elif self.score_func == "mutual_info":
            score_func = mutual_info_classif
        elif self.score_func == "f_regression":
            score_func = f_regression
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), but is: %s" % self.score_func)

        from sklearn.feature_selection import SelectKBest
        self.preprocessor = SelectKBest(score_func=score_func, k=k)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        import scipy.sparse
        import sklearn.feature_selection

        if self.preprocessor is None:
            raise NotImplementedError()

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
            raise ValueError("%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SelectKBest',
                'name': 'Select k Best',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        k = UniformIntegerHyperparameter("k", 1, 128, default_value=32)
        score_func = CategoricalHyperparameter(name="score_func",
                                               choices=["chi2", "f_classif", "mutual_info", "f_regression"],
                                               default_value="f_classif")

        cs.add_hyperparameters([score_func, k])
        return cs
