# -*- encoding: utf-8 -*-
from typing import List, Optional

import numpy as np

from scipy import sparse

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager


class XYDataManager(AbstractDataManager):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        task: int,
        feat_type: List[str],
        dataset_name: str
    ):
        super(XYDataManager, self).__init__(dataset_name)

        self.info['task'] = task
        if sparse.issparse(X):
            self.info['is_sparse'] = 1
            self.info['has_missing'] = np.all(np.isfinite(X.data))
        else:
            self.info['is_sparse'] = 0
            if hasattr(X, 'iloc'):
                self.info['has_missing'] = X.isnull().values.any()
            else:
                self.info['has_missing'] = np.all(np.isfinite(X))

        label_num = {
            REGRESSION: 1,
            BINARY_CLASSIFICATION: 2,
            MULTIOUTPUT_REGRESSION: y.shape[-1],
            MULTICLASS_CLASSIFICATION: len(np.unique(y)),
            MULTILABEL_CLASSIFICATION: y.shape[-1]
        }

        self.info['label_num'] = label_num[task]

        self.data['X_train'] = X
        self.data['Y_train'] = y
        if X_test is not None:
            self.data['X_test'] = X_test
        if y_test is not None:
            self.data['Y_test'] = y_test

        if feat_type is None:
            self.feat_type = {i: 'Numerical' for i in range(np.shape(X)[1])}
        else:
            self.feat_type = feat_type

        # TODO: try to guess task type!

        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X.shape[0],
                                                                  y.shape[0]))
