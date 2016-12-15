import copy
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.components import classification as \
    classification_components
from autosklearn.pipeline.components.choice import ClassificationComponentFilter
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing import rescaling as \
    rescaling_components
from autosklearn.pipeline.components.data_preprocessing.imputation.imputation \
    import Imputation
from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding\
    .one_hot_encoding import OneHotEncoder
from autosklearn.pipeline.components import feature_preprocessing as \
    feature_preprocessing_components
from autosklearn.pipeline.base import EstimationPipeline
from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.constants import SPARSE

class SimpleClassificationPipeline(EstimationPipeline, ClassifierMixin):

    def __init__(self, is_multiclass=False, is_multilabel=False):
        filter = ClassificationComponentFilter(is_multiclass=is_multiclass,
                                               is_multilabel=is_multilabel)

        self._output_dtype = np.int32
        components = [
                ("one_hot_encoding", OneHotEncoder()),
                ("imputation", Imputation()),
                ("rescaling", RescalingChoice(filter=filter)),
                ("preprocessor", FeaturePreprocessorChoice(filter=filter)),
                ("classifier", ClassifierChoice(filter=filter))
            ]
        super(SimpleClassificationPipeline, self).__init__(components)

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        if batch_size is None:
            Xt = X
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)

            return self.steps[-1][-1].predict_proba(Xt)

        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                        batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict_proba(X[batch_from:batch_to],
                                           batch_size=None).\
                            astype(np.float32)

                return y

