# -*- encoding: utf-8 -*-

__all__ = [
    'XYDataManager',
]
import numpy as np

from autosklearn.constants import *
from autosklearn.data_managers import SimpleDataManager
from scipy import sparse


class XYDataManager(SimpleDataManager):

    def __init__(self, data_x, y, task, metric, feat_type, dataset_name,
                 encode_labels):
        super(XYDataManager, self).__init__()
        self.info['task'] = task
        self.info['metric'] = metric
        self.info['is_sparse'] = 1 if sparse.issparse(data_x) else 0
        self.info['has_missing'] = np.all(np.isfinite(data_x))

        target_num = {
            REGRESSION: 1,
            BINARY_CLASSIFICATION: 2,
            MULTICLASS_CLASSIFICATION: len(np.unique(y)),
            MULTILABEL_CLASSIFICATION: y.shape[-1]
        }

        self.info['target_num'] = target_num[task]
        self._basename = dataset_name

        self.data['X_train'] = data_x
        self.data['Y_train'] = y
        self._feat_type = feat_type

        # TODO: try to guess task type!

        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if data_x.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (
                                 data_x.shape[0], y.shape[0]
                             ))
        if self._feat_type is None:
            self._feat_type = ['Numerical'] * data_x.shape[1]
        if data_x.shape[1] != len(self._feat_type):
            raise ValueError('X and feat type must have the same dimensions, '
                             'but are %d and %d.' %
                             (data_x.shape[1], len(self._feat_type)))

        if encode_labels:
            self.perform_hot_encoding()
