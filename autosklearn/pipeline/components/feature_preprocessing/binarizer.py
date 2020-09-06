from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter


from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA


class BinarizerComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, threshold: float = 0.0, random_state=None):
        super().__init__()
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.preprocessing import Binarizer
        self.threshold = float(self.threshold)

        self.preprocessor = Binarizer(threshold=self.threshold, copy=False)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Binarizer',
                'name': 'Binarizer',
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
        threshold = UnParametrizedHyperparameter("threshold", 0.)
        cs.add_hyperparameter(threshold)
        return cs
