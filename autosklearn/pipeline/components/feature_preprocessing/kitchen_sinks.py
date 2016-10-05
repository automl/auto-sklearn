from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *

class RandomKitchenSinks(AutoSklearnPreprocessingAlgorithm):

    def __init__(self):
        super(RandomKitchenSinks, self).__init__()
        """ Parameters:
        gamma: float
               Parameter of the rbf kernel to be approximated exp(-gamma * x^2)

        n_components: int 
               Number of components (output dimensionality) used to approximate the kernel
        """
        self.gamma = None
        self.n_components = None
        self.random_state = None

    def fit(self, X, Y=None):
        import sklearn.kernel_approximation

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.random_state)
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KitchenSink',
                'name': 'Random Kitchen Sinks',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, SIGNED_DATA),
                'output': (INPUT, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        gamma = UniformFloatHyperparameter(
            "gamma", 0.3, 2., default=1.0)
        n_components = UniformIntegerHyperparameter(
            "n_components", 50, 10000, default=100, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(n_components)
        return cs
