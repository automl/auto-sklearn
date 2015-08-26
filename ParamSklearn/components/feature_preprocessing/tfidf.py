from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration


from ..base import ParamSklearnPreprocessingAlgorithm

import numpy as np


class TFIDF(object):#ParamSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        # This is implementation is for sparse data only! It will make inplace changes to the data!

        self.idf = None
        self.random_state = random_state

    def fit(self, X, y):
        #count the number of documents in which each word occurs
        # @Stefan: Is there a reason why this is called weights and not
        # document_frequency?
        weights = (X>0.0).sum(axis=0)
        # words that never appear have to be treated differently!
        # @Stefan: Doesn't weights == 0 yield a boolean numpy array which can
        #  be directly used for indexing?
        indices = np.ravel(np.where(weights == 0)[1])
        
        # calculate (the log of) the inverse document frequencies
        self.idf = np.array(np.log(float(X.shape[0])/(weights)))[0]
        # words that are not in the training data get will be set to zero
        self.idf[indices] = 0

        return self

    def transform(self, X):
        if self.idf is None:
            raise NotImplementedError()
        X.data *= self.idf[X.indices]
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'TFIDF',
                'name': 'Term Frequency / Inverse Document Frequency',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                # TODO find out what is best used here!
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %" % name
