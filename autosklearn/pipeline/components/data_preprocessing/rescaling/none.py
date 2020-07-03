from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT, SPARSE
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm


class NoRescalingComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NoRescaling',
                'name': 'NoRescaling',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
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
                'output': (INPUT,),
                'preferred_dtype': None}
