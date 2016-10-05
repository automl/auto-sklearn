from autosklearn.pipeline.constants import *
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
# Use custom implementation because sklearn implementation cannot
# handle float32 input matrix
from autosklearn.pipeline.implementations.Normalizer import Normalizer


class NormalizerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):

    def __init__(self):
        super(NormalizerComponent, self).__init__()
        self.preprocessor = Normalizer()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Normalizer',
                'name': 'Normalizer',
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
                'input': (SPARSE, DENSE, SIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}