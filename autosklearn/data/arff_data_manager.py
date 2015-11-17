import os
import types

import arff
import numpy as np
from scipy import sparse
import six

from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.constants import *


def _load_arff(filename, target):
    with open(filename) as fh:
        decoder = arff.ArffDecoder()
        arff_object = decoder.decode(fh, encode_nominal=True)

    dataset_name = arff_object['relation']
    attributes = arff_object['attributes']
    data = arff_object['data']

    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, tuple):
        data = sparse.coo_matrix(data)
    else:
        raise ValueError('arff returned unknown data format of type %s' %
                         str(type(data)))

    target_attribute = -1
    for i, attribute in enumerate(attributes):
        if attribute[0] == target:
            target_attribute = i
            break

    if target_attribute < 0:
        raise ValueError('Target feature %s not found. Available features '
                         'are: %s' %
                         (target,
                          str([attribute[0] for attribute in attributes])))

    y = data[:,target_attribute]
    X = data[:,np.arange(data.shape[1]) != target_attribute]

    # Do not add the target to the feat_type list
    feat_type = ['Categorical' if type(attribute[1]) in (list, tuple) else
                 'Numerical' for attribute in attributes[:-1]]

    return X, y, dataset_name, feat_type


class ARFFDataManager(AbstractDataManager):
    def __init__(self, dataset, task, metric,
                 target, encode_labels=True):

        if isinstance(task, six.string_types):
            task = STRING_TO_TASK_TYPES[task]

        if isinstance(metric, six.string_types):
            metric = STRING_TO_METRIC[metric]

        train_file = os.path.join(dataset, 'train.arff')
        test_file = os.path.join(dataset, 'test.arff')

        X_train, y_train, train_relation, feat_type = _load_arff(train_file, target)
        X_test, y_test, test_relation, feat_type_test = _load_arff(test_file, target)

        super(ARFFDataManager, self).__init__(train_relation)

        self.info['task'] = task
        self.info['metric'] = metric
        self.info['is_sparse'] = 1 if sparse.issparse(X_train) else 0
        self.info['has_missing'] = not np.all(np.isfinite(X_train))

        if all([ft == 'Categorical' for ft in feat_type]):
            self.info['feat_type'] = 'Categorical'
        elif all([ft == 'Numerical' for ft in feat_type]):
            self.info['feat_type'] = 'Numerical'
        else:
            self.info['feat_type'] = 'Mixed'

        label_num = {REGRESSION: 1,
                     BINARY_CLASSIFICATION: 2,
                     MULTICLASS_CLASSIFICATION: len(np.unique(y_train)),
                     MULTILABEL_CLASSIFICATION: y_train.shape[-1]}
        self.info['label_num'] = label_num[task]
        label_num = {REGRESSION: 1,
                      BINARY_CLASSIFICATION: 1,
                      MULTICLASS_CLASSIFICATION: 1,
                      MULTILABEL_CLASSIFICATION: y_train.shape[-1]}
        self.info['label_num'] = label_num[task]

        self.data['X_train'] = X_train
        self.data['Y_train'] = y_train
        self.data['X_test'] = X_test
        self.data['Y_test'] = y_test
        self.feat_type = feat_type

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError('Train and test data must have the same number '
                         'of features, but have %d and %d.' %
                         (X_train.shape[1], X_test.shape[1]))
        if feat_type != feat_type_test:
            raise ValueError('Train and test data must have the same features.')
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X_train.shape[0],
                                                                  y_train.shape[0]))
        if self.feat_type is None:
            self.feat_type = ['Numerical'] * X_train.shape[1]
        if X_train.shape[1] != len(self.feat_type):
            raise ValueError('X and feat type must have the same dimensions, '
                             'but are %d and %d.' %
                             (X_train.shape[1], len(self.feat_type)))

        if encode_labels:
            self.perform1HotEncoding()

    # @staticmethod
    # def check_capability_of_loading_dataset(dataset):
    #     '""Checks if the ARFFDataManager is able to open the dataset
    #     given by `dataset`. It checks if the `dataset` is a directory
    #     and checks if the following files exist. If both conditions are True,
    #     the ARFFDataManager can load this dataset.
    #
    #     Necessary files:
    #
    #     * train.arff
    #     """
    #     if not os.path.exists(dataset):
    #         return False
    #     elif not os.path.isdir(dataset):
    #         return False
    #
    #     content = os.listdir(dataset)
    #     necessary_file = {'train.arff'}
    #     for dataset_file in content:
    #         try:
    #             necessary_file.remove(dataset_file)
    #         except:
    #             pass
    #
    #     if len(necessary_file) > 0:
    #         return False
    #
    #     return True


