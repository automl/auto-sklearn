import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from autosklearn.pipeline.components.data_preprocessing.category_shift.category_shift \
    import CategoryShift
from autosklearn.pipeline.components.data_preprocessing.imputation.categorical_imputation \
    import CategoricalImputation
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense.minority_coalescer \
    import MinorityCoalescer
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding.one_hot_encoding \
    import OneHotEncoder
from autosklearn.pipeline.components.data_preprocessing.rescaling.none \
    import NoRescalingComponent
from autosklearn.pipeline.components.data_preprocessing.imputation.numerical_imputation \
    import NumericalImputation
from autosklearn.pipeline.components.data_preprocessing.variance_threshold.variance_threshold \
    import VarianceThreshold


class DataPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, dtype=np.float, sparse=True, categorical_features=None):
        self.dtype = dtype
        self.sparse = sparse
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        cat_ppl = Pipeline(
            steps=[
                ('categ_shift', CategoryShift()),
                ('imputation', CategoricalImputation()),
                ('coalescense', MinorityCoalescer()),
                ('OHE', OneHotEncoder(sparse=sparse)),
            ])
        num_ppl = Pipeline(
            steps=[
                ('imputation', NumericalImputation()),
                ('var_thre', VarianceThreshold()),
                ('rescaling', NoRescalingComponent()),
            ])

        n_feats = X.shape[1]
        # If categorical_features is none or an array made just of False booleans, then
        # only the numerical transformer is used
        if self.categorical_features is None or np.all(np.logical_not(self.categorical_features)):
            sklearn_transf_spec = [
                ["numerical_transformer", num_ppl, list(range(n_feats))]
            ]
        # If all features are categorical, then just the categorical transformer is used 
        elif np.all(self.categorical_features):
            sklearn_transf_spec = [
                ["categorical_transformer", cat_ppl, list(range(n_feats))]
            ]
        # For the other cases, both transformers are used
        else:
            cat_feats = np.where(self.categorical_features)[0]
            num_feats = np.where(np.logical_not(self.categorical_features))[0]
            sklearn_transf_spec = [
                ["categorical_transformer", cat_ppl, cat_feats],
                ["numerical_transformer", num_ppl, num_feats]
            ]
        self.encoder = ColumnTransformer(sklearn_transf_spec)
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

