from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class KBinsDiscretizerComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_bins: int = 5, encode: str = "onehot", strategy: str = "quantile", random_state=None):
        super().__init__()
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.preprocessing import KBinsDiscretizer
        n_bins = int(self.n_bins)

        self.preprocessor = KBinsDiscretizer(n_bins=n_bins, encode=self.encode, strategy=self.strategy)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KBinsDiscretizer',
                'name': 'K-Bins Discretizer',
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

        n_bins = UniformIntegerHyperparameter("n_bins", 2, 100, default_value=5)
        encode = CategoricalHyperparameter("encode", ["onehot", "onehot-dense", "ordinal"], default_value="onehot")
        strategy = CategoricalHyperparameter("strategy", ["uniform", "quantile", "kmeans"], default_value="quantile")

        cs.add_hyperparameters([n_bins, encode, strategy])
        return cs
