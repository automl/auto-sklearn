# -*- encoding: utf-8 -*-
import os
import random

import numpy as np
import six

import autosklearn.automl
from autosklearn.constants import *
from autosklearn.util.backend import create
from sklearn.base import BaseEstimator


class AutoSklearnEstimator(BaseEstimator):

    def __init__(self, automl):
        self._automl = automl

    def fit(self, X, y,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None,
            ):
        pass

    def refit(self):
        pass

    def fit_ensemble(self):
        pass

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.

        """
        return self._automl.predict(X)

    def score(self):
        pass

    def show_models(self):
        return self._automl.show_models()

    def get_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'get_params() because it is not intended to '
                                  'be optimized.')

    def set_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'set_params() because it is not intended to '
                                  'be optimized.')


class AutoSklearnClassifier(AutoSklearnEstimator):

    def __init__(self, automl):
        self._classes = []
        self._n_classes = []
        self._n_outputs = []

        super(AutoSklearnClassifier, self).__init__(automl)

    def fit(self, X, y,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None,
            ):
        """Fit *autosklearn* to given training set (X, y).

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        metric : str, optional (default='acc_metric')
            The metric to optimize for. Can be one of: ['acc_metric',
            'auc_metric', 'bac_metric', 'f1_metric', 'pac_metric']. A
            description of the metrics can be found in `the paper describing
            the AutoML Challenge
            <http://www.causality.inf.ethz.ch/AutoML/automl_ijcnn15.pdf>`_.

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded.

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------
        self

        """
        # Fit is supposed to be idempotent!

        # But not if we use share_mode:
        '''
        if not self._shared_mode:
            self._delete_output_directories()
        else:
            # If this fails, it's likely that this is the first call to get
            # the data manager
            try:
                D = self._backend.load_datamanager()
                dataset_name = D.name
            except IOError:
                pass

        self._create_output_directories()

        '''
        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self._n_outputs = y.shape[1]

        y = np.copy(y)

        self._classes = []
        self._n_classes = []

        for k in six.moves.range(self._n_outputs):
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

        return self._automl.fit(X, y, task, metric, feat_type, dataset_name)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        max_probability_index = np.argmax(probabilities, axis=1)

        return max_probability_index

    def predict_proba(self, X):
        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples, n_classes] or [n_samples, n_labels]
            The predicted class probabilities.
        """
        return self._automl.predict(X)


class AutoSklearnRegressor(AutoSklearnEstimator):

    def fit(self, X, y,
            metric='r2_metric',
            feat_type=None,
            dataset_name=None,
            ):

        task = REGRESSION

        return self._automl.fit(X, y, task, metric, feat_type, dataset_name)


class EstimatorBuilder():
    """This class implements the classification task.

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

    include_estimators : dict, optional (None)
        If None all possible estimators are used. Otherwise specifies set of
        estimators to use

    include_preprocessors : dict, optional (None)
        If None all possible preprocessors are used. Otherwise specifies set of
        preprocessors to use

    resampling_strategy : string, optional ('holdout')
        how to to handle overfitting, might need 'resampling_strategy_arguments'

        * 'holdout': 66:33 (train:test) split
        * 'holdout-iterative-fit':  66:33 (train:test) split, calls iterative
          fit where possible
        * 'cv': crossvalidation, requires 'folds'
        * 'nested-cv': crossvalidation, requires 'outer-folds, 'inner-folds'
        * 'partial-cv': crossvalidation, requires 'folds' , calls
          iterative fit where possible

    resampling_strategy_arguments : dict, optional if 'holdout' (None)
        Additional arguments for resampling_strategy
        * 'holdout': None
        * 'holdout-iterative-fit':  None
        * 'cv': {'folds': int}
        * 'nested-cv': {'outer_folds': int, 'inner_folds'
        * 'partial-cv': {'folds': int}

    tmp_folder : string, optional (None)
        folder to store configuration output, if None automatically use
        /tmp/autosklearn_tmp_$pid_$random_number

    output_folder : string, optional (None)
        folder to store trained models, if None automatically use
        /tmp/autosklearn_output_$pid_$random_number

    delete_tmp_folder_after_terminate: string, optional (True)
        remove tmp_folder, when finished. If tmp_folder is None
        tmp_dir will always be deleted

    delete_output_folder_after_terminate: bool, optional (True)
        remove output_folder, when finished. If output_folder is None
        output_dir will always be deleted

    shared_mode: bool, optional (False)
        run smac in shared-model-node. This only works if arguments
        tmp_folder and output_folder are given and sets both
        delete_tmp_folder_after_terminate and
        delete_output_folder_after_terminate to False.

    """
    @staticmethod
    def __new__(cls,
               time_left_for_this_task=3600,
               per_run_time_limit=360,
               initial_configurations_via_metalearning=25,
               ensemble_size=50,
               ensemble_nbest=50,
               seed=1,
               ml_memory_limit=3000,
               include_estimators=None,
               include_preprocessors=None,
               resampling_strategy='holdout',
               resampling_strategy_arguments=None,
               tmp_folder=None,
               output_folder=None,
               delete_tmp_folder_after_terminate=True,
               delete_output_folder_after_terminate=True,
               shared_mode=False):

        estimator_cls = next(x for x in cls.__bases__ if issubclass(x, AutoSklearnEstimator))

        if shared_mode:
            delete_output_folder_after_terminate = False
            delete_tmp_folder_after_terminate = False
            if tmp_folder is None:
                raise ValueError("If shared_mode == True tmp_folder must not "
                                 "be None.")
            if output_folder is None:
                raise ValueError("If shared_mode == True output_folder must "
                                 "not be None.")

        backend = create(tmp_folder, output_folder, delete_tmp_folder_after_terminate, delete_output_folder_after_terminate)
        automl = autosklearn.automl.AutoML(
            backend=backend,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            log_dir=backend.temporary_directory,
            initial_configurations_via_metalearning=
            initial_configurations_via_metalearning,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            seed=seed,
            ml_memory_limit=ml_memory_limit,
            include_estimators=include_estimators,
            include_preprocessors=include_preprocessors,
            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_strategy_arguments,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=
            delete_output_folder_after_terminate,
            shared_mode=shared_mode)
        estimator = estimator_cls(automl)

        return estimator

