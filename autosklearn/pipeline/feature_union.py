from ConfigSpace import ConfigurationSpace

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm, AutoSklearnComponent
from autosklearn.pipeline.constants import DENSE, SPARSE
from autosklearn.pipeline.constants import INPUT, SIGNED_DATA


class FeatureUnion(AutoSklearnComponent):

    def __init__(self, components_list, random_state=None):
        super(FeatureUnion, self).__init__()
        self.components_list = components_list

    def fit(self, X, y=None):
        import sklearn.pipeline
        self.preprocessor = sklearn.pipeline.make_union(self.components_list)
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FeatureUnion',
                'name': 'Feature Union',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (SIGNED_DATA),
                'output': (INPUT, )}

    def get_hyperparameter_search_space(self, dataset_properties=None):
        cs = ConfigurationSpace()

        for idx, component in enumerate(self.components_list):
            sub_cs = component.get_hyperparameter_search_space()
            sub_cs = None

        return cs


    def set_hyperparameters(self, configuration):
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        return self
