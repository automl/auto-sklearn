from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class CategoricalImputation(AutoSklearnPreprocessingAlgorithm):
    """ Imputation of categorical features. By default, replace missing values by the 
    integer 2.

    Parameters
    ----------
    strategy : str, optional
        Substitution strategy. Shoudl be either ''constant' or 'most_frequent', 
        by default 'constant'
    fill_value : int, optional
        Substitution value in case strategy='constant', by default 2
    random_state : [type], optional
        [description], by default None
    
    """

    def __init__(self, strategy='constant', fill_value=2, random_state=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        import sklearn.impute

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy=self.strategy, fill_value=self.fill_value, copy=False)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CategImputation',
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
