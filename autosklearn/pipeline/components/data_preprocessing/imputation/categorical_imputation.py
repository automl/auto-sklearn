from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class CategoricalImputation(AutoSklearnPreprocessingAlgorithm):
    """
    Substitute missing values by 2
    """

    def __init__(self, random_state=None):
        self.random_stated = random_state

    def fit(self, X, y=None):
        import sklearn.impute

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy='constant', fill_value=2, copy=False)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X).astype(int)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CategoricalImputation',
                'name': 'Categorical Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()
