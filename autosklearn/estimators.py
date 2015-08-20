# -*- encoding: utf-8 -*-
import os
import random
import shutil

import numpy as np

import autosklearn.automl
from autosklearn.constants import *


class AutoSklearnClassifier(autosklearn.automl.AutoML):

    """This class implements the classification task. It must not be pickled!

    Parameters
    ----------
    time_left_for_this_task : int, optional (default=3600)
        Time limit in seconds for the search for appropriate classification
        models. By increasing this value, *auto-sklearn* will find better
        configurations.

    per_run_time_limit : int, optional (default=360)
        Time limit for a single call to machine learning model.

    initial_configurations_via_metalearning : int, optional (default=25)

    ensemble_size : int, optional (default=50)

    ensemble_nbest : int, optional (default=50)

    seed : int, optional (default=1)

    ml_memory_limit : int, optional (3000)
        Memory limit for the machine learning algorithm. If the machine
        learning algorithm allocates tries to allocate more memory,
        its evaluation will be stopped.

    """

    def __init__(self,
                 time_left_for_this_task=3600,
                 per_run_time_limit=360,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=50,
                 ensemble_nbest=50,
                 seed=1,
                 ml_memory_limit=3000,
                 tmp_folder=None,
                 output_folder=None):

        self._tmp_dir, self._output_dir = self._prepare_create_folders(
            tmp_folder,
            output_folder
        )

        self._classes = []
        self._n_classes = []
        self._n_outputs = []

        super(AutoSklearnClassifier, self).__init__(
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            tmp_dir=self._tmp_dir,
            output_dir=self._output_dir,
            log_dir=self._tmp_dir,
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            seed=seed,
            ml_memory_limit=ml_memory_limit)

    @staticmethod
    def _prepare_create_folders(tmp_dir, output_dir):
        random_number = random.randint(0, 10000)
        pid = os.getpid()
        if tmp_dir is None:
            tmp_dir = '/tmp/autosklearn_tmp_%d_%d' % (pid, random_number)
        if output_dir is None:
            output_dir = '/tmp/autosklearn_output_%d_%d' % (pid, random_number)
        os.makedirs(output_dir)
        os.makedirs(tmp_dir)
        return tmp_dir, output_dir

    def __del__(self):
        self._delete_output_directories()

    def _create_output_directories(self):
        os.makedirs(self._output_dir)
        os.makedirs(self._tmp_dir)

    def _delete_output_directories(self):
        shutil.rmtree(self._tmp_dir)
        shutil.rmtree(self._output_dir)

    def fit(self, data_x, y,
            task=MULTICLASS_CLASSIFICATION,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None,
            ):
        """Fit *autosklearn* to given training set (X, y).

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        metric : str, optional (default='acc_metric')
            The metric to optimize for. Can be one of: ['acc_metric',
            'auc_metric', 'bac_metric', 'f1_metric', 'pac_metric']

        feat_type : list, optional (default=None)
            List of :python:`len(X.shape[1])` describing if an attribute is
            continuous or categorical. Categorical attributes will
            automatically 1Hot encoded.
        """
        # Fit is supposed to be idempotent!
        self._delete_output_directories()
        self._create_output_directories()

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self._n_outputs = y.shape[1]

        y = np.copy(y)

        self._classes = []
        self._n_classes = []

        for k in xrange(self._n_outputs):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self._classes.append(classes_k)
            self._n_classes.append(classes_k.shape[0])

        self._n_classes = np.array(self._n_classes, dtype=np.int)

        if self._n_outputs > 1:
            task = MULTILABEL_CLASSIFICATION
        else:
            if len(self._classes[0]) == 2:
                task = BINARY_CLASSIFICATION
            else:
                task = MULTICLASS_CLASSIFICATION

        # TODO: fix metafeatures calculation to allow this!
        if y.shape[1] == 1:
            y = y.flatten()

        return super(AutoSklearnClassifier, self).fit(data_x, y, task, metric,
                                                      feat_type, dataset_name)

    def predict(self, data_x):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.

        """
        return super(AutoSklearnClassifier, self).predict(data_x)


class AutoSklearnRegressor(autosklearn.automl.AutoML):

    def __init__(self, **kwargs):
        raise NotImplementedError()
