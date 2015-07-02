import numpy as np
from scipy import sparse

from autosklearn.data.data_manager import DataManager


class XyDataManager(DataManager):
    def __init__(self, X, y, task, metric, feat_type, dataset_name,
                 encode_labels):
        super(XyDataManager, self).__init__()
        self.info['task'] = task
        self.info['metric'] = metric
        self.info['is_sparse'] = 1 if sparse.issparse(X) else 0
        self.info['has_missing'] = np.all(np.isfinite(X))

        target_num = {'regression': 1,
                      'binary.classification': 2,
                      'multiclass.classification': len(np.unique(y)),
                      'multilabel.classification': y.shape[-1]}

        self.info['target_num'] = target_num[task]
        self.basename = dataset_name

        self.data["X_train"] = X
        self.data["Y_train"] = y
        self.feat_type = feat_type

        # TODO: try to guess task type!

        if len(y.shape) > 2:
            raise ValueError("y must not have more than two dimensions, "
                             "but has %d." % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of "
                             "datapoints, but have %d and %d." % (X.shape[0],
                                                                  y.shape[0]))
        if self.feat_type is None:
            self.feat_type = ['Numerical'] * X.shape[0]
        if X.shape[0] != len(self.feat_type):
            raise ValueError("X and feat type must have the same dimensions, "
                             "but are %d and %d." %
                             (X.shape[1], len(self.feat_type)))

        if encode_labels:
            self.perform1HotEncoding()



