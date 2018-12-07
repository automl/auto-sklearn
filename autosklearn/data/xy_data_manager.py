# -*- encoding: utf-8 -*-

import numpy as np
from scipy import sparse

from autosklearn.constants import *
from autosklearn.data.abstract_data_manager import AbstractDataManager


class XYDataManager(AbstractDataManager):

    def __init__(self, X, y, X_test, y_test, task, feat_type, dataset_name):
        super(XYDataManager, self).__init__(dataset_name)

        if isinstance(task, str):
            task = STRING_TO_TASK_TYPES[task]

        self.info['task'] = task
        if sparse.issparse(X):
            self.info['is_sparse'] = 1
            self.info['has_missing'] = np.all(np.isfinite(X.data))
        else:
            self.info['is_sparse'] = 0
            self.info['has_missing'] = np.all(np.isfinite(X))

        label_num = {
            REGRESSION: 1,
            BINARY_CLASSIFICATION: 2,
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

        if feat_type is not None:
            for feat in feat_type:
                allowed_types = ['numerical', 'categorical']
                if feat.lower() not in allowed_types:
                    raise ValueError("Entry '%s' in feat_type not in %s" %
                                     (feat.lower(), str(allowed_types)))

        self.feat_type = feat_type

        # TODO: try to guess task type!

        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X.shape[0],
                                                                  y.shape[0]))
        if self.feat_type is None:
            self.feat_type = ['Numerical'] * X.shape[1]
        if X.shape[1] != len(self.feat_type):
            raise ValueError('X and feat type must have the same dimensions, '
                             'but are %d and %d.' %
                             (X.shape[1], len(self.feat_type)))
