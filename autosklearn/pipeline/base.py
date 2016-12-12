from abc import ABCMeta

import numpy as np
from ConfigSpace import Configuration
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state, check_is_fitted

from autosklearn.pipeline.components.serial import SerialAutoSklearnComponent

class EstimationPipeline(SerialAutoSklearnComponent, Pipeline):

    def __init__(self, components):
        SerialAutoSklearnComponent.__init__(self, components)
        Pipeline.__init__(self, self.components.items())

    def fit(self, X, y, fit_params=None, init_params=None):
        X, fit_params = self.pre_transform(X, y, fit_params=fit_params)
        self.fit_estimator(X, y, **fit_params)
        return self

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        # TODO do something with the init params!
        # TODO actually, initialize the submodels only here?
        if fit_params is None or not isinstance(fit_params, dict):
            fit_params = dict()
        else:
            fit_params = {key.replace(":", "__"): value for key, value in
                          fit_params.items()}
        X, fit_params = self._pre_transform(X, y, **fit_params)
        return X, fit_params

    def fit_estimator(self, X, y, **fit_params):
        if fit_params is None:
            fit_params = {}
        self.steps[-1][-1].fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        if fit_params is None:
            fit_params = {}
        self.steps[-1][-1].iterative_fit(X, y, n_iter=n_iter,
                                         **fit_params)

    def estimator_supports_iterative_fit(self):
        return hasattr(self.steps[-1][-1], 'iterative_fit')

    def configuration_fully_fitted(self):
        check_is_fitted(self, 'pipeline_')
        return self.steps[-1][-1].configuration_fully_fitted()

    def predict(self, X, batch_size=None):
        if batch_size is None:
            return super(EstimationPipeline, self).predict(X).astype(self._output_dtype)
        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],), dtype=self._output_dtype)
                else:
                    y = np.zeros((X.shape[0], self.num_targets),
                                 dtype=self._output_dtype)

                # Copied and adapted from the scikit-learn GP code
                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                                                          batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to], batch_size=None)

                return y

    def get_hyperparameter_search_space(self, include=None, exclude=None,
                                        dataset_properties=None):
        return self.get_config_space()
