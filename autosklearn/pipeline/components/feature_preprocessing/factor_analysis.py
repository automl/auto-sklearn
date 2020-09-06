from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA


class FactorAnalysisComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_components: float = None,
                 max_iter: int = 1000,
                 svd_method: str = "randomized",
                 iterated_power: int = 3,
                 tol: float = 1e-2,
                 random_state=None
                 ):
        super().__init__()
        self.n_components = n_components
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.decomposition import FactorAnalysis
        self.n_components = int(self.n_components)
        self.tol = float(self.tol)
        self.max_iter = int(self.max_iter)
        self.iterated_power = int(self.iterated_power)

        self.preprocessor = FactorAnalysis(n_components=self.n_components,
                                           svd_method=self.svd_method,
                                           max_iter=self.max_iter,
                                           iterated_power=self.iterated_power,
                                           tol=self.tol,
                                           random_state=self.random_state,
                                           copy=False)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FactorAnalysis',
                'name': 'FactorAnalysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
        max_iter = UniformIntegerHyperparameter("max_iter", 10, 2000, default_value=1000)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-2, log=True)
        svd_method = CategoricalHyperparameter("svd_method", ["lapack", "randomized"], default_value="randomized")
        iterated_power = UniformIntegerHyperparameter("iterated_power", 1, 10, default_value=3)
        cs.add_hyperparameters([n_components, max_iter, tol, svd_method, iterated_power])

        iterated_power_condition = InCondition(iterated_power, svd_method, ["randomized"])
        cs.add_condition(iterated_power_condition)

        return cs
