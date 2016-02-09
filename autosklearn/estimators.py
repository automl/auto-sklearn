# -*- encoding: utf-8 -*-
import os
import random

import numpy as np
import six

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

    def __init__(self,
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

        # Check this before _prepare_create_folders assigns random output
        # directories
        if shared_mode:
            delete_output_folder_after_terminate = False
            delete_tmp_folder_after_terminate = False
            if tmp_folder is None:
                raise ValueError("If shared_mode == True tmp_folder must not "
                                 "be None.")
            if output_folder is None:
                raise ValueError("If shared_mode == True output_folder must "
                                 "not be None.")

        # Call this before calling superconstructor as we feed tmp/output dir
        # to superinit
        self._tmp_dir, self._output_dir = self._prepare_create_folders(
            tmp_dir=tmp_folder,
            output_dir=output_folder)

        self._classes = []
        self._n_classes = []
        self._n_outputs = []

        super(AutoSklearnClassifier, self).__init__(
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            log_dir=self._tmp_dir,
            initial_configurations_via_metalearning=
            initial_configurations_via_metalearning,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            seed=seed,
            ml_memory_limit=ml_memory_limit,
            include_estimators=include_estimators,
            include_preprocessors=include_preprocessors,
            resampling_strategy=resampling_strategy,
            tmp_dir=self._tmp_dir,
            output_dir=self._output_dir,
            resampling_strategy_arguments=resampling_strategy_arguments,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=
            delete_output_folder_after_terminate,
            shared_mode=shared_mode)

    @staticmethod
    def _prepare_create_folders(tmp_dir, output_dir):
        random_number = random.randint(0, 10000)

        pid = os.getpid()
        if tmp_dir is None:
            tmp_dir = '/tmp/autosklearn_tmp_%d_%d' % (pid, random_number)
        if output_dir is None:
            output_dir = '/tmp/autosklearn_output_%d_%d' % (pid, random_number)

        # Totally weird, this has to be created here, will be deleted in the
        # first lines of fit(). If not there, creating the Backend object in the
        # superclass will fail
        try:
            os.makedirs(tmp_dir)
        except OSError:
            pass
        try:
            os.makedirs(output_dir)
        except OSError:
            pass

        return tmp_dir, output_dir

    def _create_output_directories(self):
        try:
            os.makedirs(self._tmp_dir)
        except OSError:
            pass
        try:
            os.makedirs(self._output_dir)
        except OSError:
            pass

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

        return super(AutoSklearnClassifier, self).fit(X, y, task, metric,
                                                      feat_type, dataset_name)

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
        return super(AutoSklearnClassifier, self).predict(X)

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
        return super(AutoSklearnClassifier, self).predict_proba(X)


class AutoSklearnRegressor(autosklearn.automl.AutoML):

    def __init__(self, **kwargs):
        raise NotImplementedError()
