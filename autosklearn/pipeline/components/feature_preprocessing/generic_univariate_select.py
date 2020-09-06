from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter
from sklearn import feature_selection

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class GenericUnivariateSelectComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, param: float = 1e-05, score_func: str = "f_classif", mode: str = "percentile",
                 random_state=None):
        super().__init__()
        self.param = param
        self.mode = mode
        self.score_func = score_func
        self.random_state = random_state

        self.score_func_mapping = dict(chi2=feature_selection.chi2,
                                       f_classif=feature_selection.f_classif,
                                       f_regression=feature_selection.f_regression)

    def fit(self, X, Y=None):
        from sklearn.feature_selection import GenericUnivariateSelect

        if isinstance(self.score_func, str):
            score_func = self.score_func_mapping[self.score_func]
        else:
            score_func = self.score_func

        self.preprocessor = GenericUnivariateSelect(param=self.param, mode=self.mode, score_func=score_func)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GenericUnivariateSelect',
                'name': 'Generic Univariate Select',
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

        mode = CategoricalHyperparameter("mode", ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'],
                                         default_value="percentile")
        param = UniformFloatHyperparameter("param", 1e-05, 0.75, default_value=1e-05)
        score_func = CategoricalHyperparameter(name="score_func", choices=["chi2", "f_classif", "f_regression"],
                                               default_value="f_classif")

        cs.add_hyperparameters([mode, param, score_func])
        return cs
