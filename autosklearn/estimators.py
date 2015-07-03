import os
import random
import shutil

import numpy as np

import autosklearn.automl
from autosklearn.constants import *


class AutoSklearnClassifier(autosklearn.automl.AutoML):
    def __init__(self, time_left_for_this_task=3600,
                 per_run_time_limit=360,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1, ensemble_nbest=1, seed=1,
                 ml_memory_limit=3000):
        random_number = random.randint(0, 10000)

        pid = os.getpid()
        output_dir = "/tmp/autosklearn_output_%d_%d" % (pid, random_number)
        tmp_dir = "/tmp/autosklearn_tmp_%d_%d" % (pid, random_number)
        os.makedirs(output_dir)
        os.makedirs(tmp_dir)

        super(AutoSklearnClassifier, self).__init__(
            tmp_dir, output_dir, time_left_for_this_task, per_run_time_limit,
            log_dir=tmp_dir,
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            ensemble_size=ensemble_size, ensemble_nbest=ensemble_nbest,
            seed=seed, ml_memory_limit=ml_memory_limit)

    def __del__(self):
        self._delete_output_directories()

    def _create_output_directories(self):
        os.makedirs(self.output_dir)
        os.makedirs(self.tmp_dir)

    def _delete_output_directories(self):
        shutil.rmtree(self.tmp_dir)
        shutil.rmtree(self.output_dir)

    def fit(self, X, y, metric='acc_metric', feat_type=None):
        # Fit is supposed to be idempotent!
        self._delete_output_directories()
        self._create_output_directories()

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        for k in xrange(self.n_outputs_):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        self.n_classes_ = np.array(self.n_classes_, dtype=np.int)

        if self.n_outputs_ > 1:
            task = MULTILABEL_CLASSIFICATION
        else:
            if len(self.classes_[0]) == 2:
                task = BINARY_CLASSIFICATION
            else:
                task = MULTICLASS_CLASSIFICATION

        # TODO: fix metafeatures calculation to allow this!
        if y.shape[1] == 1:
            y = y.flatten()

        return super(AutoSklearnClassifier, self).fit(X, y, task, metric,
                                                   feat_type)


class AutoSklearnRegressor(autosklearn.automl.AutoML):
    def __init__(self, **kwargs):
        raise NotImplementedError()