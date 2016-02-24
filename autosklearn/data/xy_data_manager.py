# -*- encoding: utf-8 -*-
import types

import numpy as np
from scipy import sparse
import six

from autosklearn.constants import *
from autosklearn.data.abstract_data_manager import AbstractDataManager


class XYDataManager(AbstractDataManager):

    def __init__(self, data_x, y, task, metric, feat_type, dataset_name,
                 encode_labels):
        super(XYDataManager, self).__init__(dataset_name)

        if isinstance(task, six.string_types):
            task = STRING_TO_TASK_TYPES[task]

        if isinstance(metric, six.string_types):
            metric = STRING_TO_METRIC[metric]

        self.info['task'] = task
        self.info['metric'] = metric
        if sparse.issparse(data_x):
            self.info['is_sparse'] = 1
            self.info['has_missing'] = np.all(np.isfinite(data_x.data))
        else:
            self.info['is_sparse'] = 0
            self.info['has_missing'] = np.all(np.isfinite(data_x))

        label_num = {
            REGRESSION: 1,
            BINARY_CLASSIFICATION: 2,
            MULTICLASS_CLASSIFICATION: len(np.unique(y)),
            MULTILABEL_CLASSIFICATION: y.shape[-1]
        }

        self.info['label_num'] = label_num[task]

        self.data['X_train'] = data_x
        self.data['Y_train'] = y
        self.feat_type = feat_type

        # TODO: try to guess task type!

        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if data_x.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (data_x.shape[0],
                                                                  y.shape[0]))
        if self.feat_type is None:
            self.feat_type = ['Numerical'] * data_x.shape[1]
        if data_x.shape[1] != len(self.feat_type):
            raise ValueError('X and feat type must have the same dimensions, '
                             'but are %d and %d.' %
                             (data_x.shape[1], len(self.feat_type)))

        if encode_labels:
            self.perform1HotEncoding()
