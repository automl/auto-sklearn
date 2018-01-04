from autosklearn.pipeline.constants import *
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm


class MinMaxScalerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state):
        from sklearn.preprocessing import MinMaxScaler
        self.preprocessor = MinMaxScaler(copy=False)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MinMaxScaler',
                'name': 'MinMaxScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}