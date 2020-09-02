from scipy import sparse
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, SIGNED_DATA, INPUT, SPARSE
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm


class RobustScalerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(self, q_min, q_max, random_state):
        from sklearn.preprocessing import RobustScaler
        self.q_min = q_min
        self.q_max = q_max
        self.preprocessor = RobustScaler(
            quantile_range=(self.q_min, self.q_max), copy=False,
        )

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RobustScaler',
                'name': 'RobustScaler',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                # TODO find out if this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}

    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter(
            'q_min', 0.001, 0.3, default_value=0.25
        )
        q_max = UniformFloatHyperparameter(
            'q_max', 0.7, 0.999, default_value=0.75
        )
        cs.add_hyperparameters((q_min, q_max))
        return cs

    def fit(self, X, y=None):
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_centering=False)

        return super(RobustScalerComponent, self).fit(X, y)
