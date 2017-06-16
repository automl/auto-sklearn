# -*- encoding: utf-8 -*-

import numpy as np
import warnings

from sklearn.metrics.classification import type_of_target
from sklearn.base import BaseEstimator
import sklearn.utils
import scipy.sparse

import autosklearn.automl
from autosklearn.metrics import f1_macro, accuracy, r2
from autosklearn.constants import *
from autosklearn.util.backend import create


class AutoMLDecorator(object):

    def __init__(self, automl):
        self._automl = automl

    def fit(self, *args, **kwargs):
        self._automl.fit(*args, **kwargs)

    def refit(self, X, y):
        """Refit all models found with fit to new data.

        Necessary when using cross-validation. During training, auto-sklearn
        fits each model k times on the dataset, but does not keep any trained
        model and can therefore not be used to predict for new data points.
        This methods fits all models found during a call to fit on the data
        given. This method may also be used together with holdout to avoid
        only using 66% of the training data to fit the final model.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The targets.

        Returns
        -------

        self

        """
        return self._automl.refit(X, y)

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):

        return self._automl.fit_ensemble(y, task, metric, precision,
                                         dataset_name, ensemble_nbest,
                                         ensemble_size)

    def predict(self, X, batch_size=None, n_jobs=1):
        return self._automl.predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, X, y):
        return self._automl.score(X, y)

    def show_models(self):
        """Return a representation of the final ensemble found by auto-sklearn.

        Returns
        -------
        str

        """
        return self._automl.show_models()

    @property
    def cv_results_(self):
        return self._automl.cv_results_

    @property
    def trajectory_(self):
        return self._automl.trajectory_

    @property
    def fANOVA_input_(self):
        return self._automl.fANOVA_input_

    def sprint_statistics(self):
        return self._automl.sprint_statistics()


class AutoSklearnEstimator(AutoMLDecorator, BaseEstimator):

    def __init__(self,
                 time_left_for_this_task=3600,
                 per_run_time_limit=360,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=50,
                 ensemble_nbest=50,
                 seed=1,
                 ml_memory_limit=3072,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 resampling_strategy='holdout',
                 resampling_strategy_arguments=None,
                 tmp_folder=None,
                 output_folder=None,
                 delete_tmp_folder_after_terminate=True,
                 delete_output_folder_after_terminate=True,
                 shared_mode=False,
                 disable_evaluator_output=False,
                 configuration_mode='SMAC'):
        """
        Parameters
        ----------
        time_left_for_this_task : int, optional (default=3600)
            Time limit in seconds for the search of appropriate
            models. By increasing this value, *auto-sklearn* has a higher
            chance of finding better models.


        per_run_time_limit : int, optional (default=360)
            Time limit for a single call to the machine learning model.
            Model fitting will be terminated if the machine learning
            algorithm runs over the time limit. Set this value high enough so
            that typical machine learning algorithms can be fit on the
            training data.

        initial_configurations_via_metalearning : int, optional (default=25)
            Initialize the hyperparameter optimization algorithm with this
            many configurations which worked well on previously seen
            datasets. Disable if the hyperparameter optimization algorithm
            should start from scratch.

        ensemble_size : int, optional (default=50)
            Number of models added to the ensemble built by *Ensemble
            selection from libraries of models*. Models are drawn with
            replacement.

        ensemble_nbest : int, optional (default=50)
            Only consider the ``ensemble_nbest`` models when building an
            ensemble. Implements `Model Library Pruning` from `Getting the
            most out of ensemble selection`.

        seed : int, optional (default=1)

        ml_memory_limit : int, optional (3072)
            Memory limit in MB for the machine learning algorithm.
            `auto-sklearn` will stop fitting the machine learning algorithm if
            it tries to allocate more than `ml_memory_limit` MB.

        include_estimators : list, optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators to use.

        exclude_estimators : list, optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators not to use. Incompatible with include_estimators.

        include_preprocessors : list, optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors to use.

        exclude_preprocessors : list, optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors not to use. Incompatible with
            include_preprocessors.

        resampling_strategy : string, optional ('holdout')
            how to to handle overfitting, might need 'resampling_strategy_arguments'

            * 'holdout': 67:33 (train:test) split
            * 'holdout-iterative-fit':  67:33 (train:test) split, calls iterative
              fit where possible
            * 'cv': crossvalidation, requires 'folds'

        resampling_strategy_arguments : dict, optional if 'holdout' (train_size default=0.67)
            Additional arguments for resampling_strategy
            ``train_size`` should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split.
            * 'holdout': {'train_size': float}
            * 'holdout-iterative-fit':  {'train_size': float}
            * 'cv': {'folds': int}

        tmp_folder : string, optional (None)
            folder to store configuration output and log files, if ``None``
            automatically use ``/tmp/autosklearn_tmp_$pid_$random_number``

        output_folder : string, optional (None)
            folder to store predictions for optional test set, if ``None``
            automatically use ``/tmp/autosklearn_output_$pid_$random_number``

        delete_tmp_folder_after_terminate: string, optional (True)
            remove tmp_folder, when finished. If tmp_folder is None
            tmp_dir will always be deleted

        delete_output_folder_after_terminate: bool, optional (True)
            remove output_folder, when finished. If output_folder is None
            output_dir will always be deleted

        shared_mode: bool, optional (False)
            Run smac in shared-model-node. This only works if arguments
            ``tmp_folder`` and ``output_folder`` are given and both
            ``delete_tmp_folder_after_terminate`` and
            ``delete_output_folder_after_terminate`` are set to False.

        disable_evaluator_output: bool or list, optional (False)
            If True, disable model and prediction output. Cannot be used
            together with ensemble building. ``predict()`` cannot be used when
            setting this True. Can also be used as a list to pass more
            fine-grained information on what to save. Allowed elements in the
            list are:

            * ``'y_optimization'`` : do not save the predictions for the
              optimization/validation set, which would later on be used to build
              an ensemble.
            * ``'model'`` : do not save any model files

        configuration_mode : ``SMAC`` or ``ROAR``
            Defines the configuration mode as described in the paper
            `Sequential Model-Based Optimization for General Algorithm
            Configuration <http://aad.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf>`_:

            * ``SMAC`` (default): Sequential Model-based Algorithm
              Configuration, which is a Bayesian optimization algorithm
            * ``ROAR``: Random Online Aggressive Racing, which is basically
              random search

        Attributes
        ----------

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            Not all keys returned by scikit-learn are supported yet.

        """
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.ml_memory_limit = ml_memory_limit
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_preprocessors = include_preprocessors
        self.exclude_preprocessors = exclude_preprocessors
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_arguments = resampling_strategy_arguments
        self.tmp_folder = tmp_folder
        self.output_folder = output_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        self.shared_mode = shared_mode
        self.disable_evaluator_output = disable_evaluator_output
        self.configuration_mode = configuration_mode
        super(AutoSklearnEstimator, self).__init__(None)

    def build_automl(self):
        if self.shared_mode:
            self.delete_output_folder_after_terminate = False
            self.delete_tmp_folder_after_terminate = False
            if self.tmp_folder is None:
                raise ValueError("If shared_mode == True tmp_folder must not "
                                 "be None.")
            if self.output_folder is None:
                raise ValueError("If shared_mode == True output_folder must "
                                 "not be None.")

        backend = create(temporary_directory=self.tmp_folder,
                         output_directory=self.output_folder,
                         delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
                         delete_output_folder_after_terminate=self.delete_output_folder_after_terminate)
        automl = autosklearn.automl.AutoML(
            backend=backend,
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            log_dir=backend.temporary_directory,
            initial_configurations_via_metalearning=
            self.initial_configurations_via_metalearning,
            ensemble_size=self.ensemble_size,
            ensemble_nbest=self.ensemble_nbest,
            seed=self.seed,
            ml_memory_limit=self.ml_memory_limit,
            include_estimators=self.include_estimators,
            exclude_estimators=self.exclude_estimators,
            include_preprocessors=self.include_preprocessors,
            exclude_preprocessors=self.exclude_preprocessors,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_arguments=self.resampling_strategy_arguments,
            delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=
            self.delete_output_folder_after_terminate,
            shared_mode=self.shared_mode,
            configuration_mode=self.configuration_mode,
            disable_evaluator_output=self.disable_evaluator_output)

        return automl

    def fit(self, *args, **kwargs):
        self._automl = self.build_automl()
        super(AutoSklearnEstimator, self).fit(*args, **kwargs)

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        """Fit an ensemble to models trained during an optimization process.

        All parameters are ``None`` by default. If no other value is given,
        the default values which were set in a call to ``fit()`` are used.

        Parameters
        ----------
        y : array-like
            Target values.

        task : int
            A constant from the module ``autosklearn.constants``. Determines
            the task type (binary classification, multiclass classification,
            multilabel classification or regression).

        metric : callable, optional
            An instance of :class:`autosklearn.metrics.Scorer` as created by
            :meth:`autosklearn.metrics.make_scorer`. These are the `Built-in
            Metrics`_.

        precision : str
            Numeric precision used when loading ensemble data. Can be either
            ``'16'``, ``'32'`` or ``'64'``.

        dataset_name : str
            Name of the current data set.

        ensemble_nbest : int
            Determines how many models should be considered from the ensemble
            building. This is inspired by a concept called library pruning
            introduced in `Getting Most out of Ensemble Selection`.

        ensemble_size : int
            Size of the ensemble built by `Ensomble Selection`.

        Returns
        -------
        self

        """
        if self._automl is None:
            self._automl = self.build_automl()
        return self._automl.fit_ensemble(y, task, metric, precision,
                                         dataset_name, ensemble_nbest,
                                         ensemble_size)


class AutoSklearnClassifier(AutoSklearnEstimator):
    """
    This class implements the classification task.

    """

    def build_automl(self):
        automl = super(AutoSklearnClassifier, self).build_automl()
        return AutoMLClassifier(automl)

    def fit(self, X, y,
            metric=None,
            feat_type=None,
            dataset_name=None):
        """Fit *auto-sklearn* to given training set (X, y).

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        metric : callable, optional (default='autosklearn.metrics.accuracy')
            An instance of :class:`autosklearn.metrics.Scorer` as created by
            :meth:`autosklearn.metrics.make_scorer`. These are the `Built-in
            Metrics`_.

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------
        self

        """
        return super(AutoSklearnClassifier, self).fit(X=X, y=y, metric=metric,
                                                      feat_type=feat_type,
                                                      dataset_name=dataset_name)

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.

        """
        return super(AutoSklearnClassifier, self).predict(
            X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):

        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples, n_classes] or [n_samples, n_labels]
            The predicted class probabilities.

        """
        return self._automl.predict_proba(
            X, batch_size=batch_size, n_jobs=n_jobs)


class AutoSklearnRegressor(AutoSklearnEstimator):
    """
    This class implements the regression task.

    """

    def build_automl(self):
        automl = super(AutoSklearnRegressor, self).build_automl()
        return AutoMLRegressor(automl)

    def fit(self, X, y,
            metric=None,
            feat_type=None,
            dataset_name=None):
        """Fit *autosklearn* to given training set (X, y).

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The regression target.

        metric : callable, optional (default='autosklearn.metrics.r2')
            An instance of :class:`autosklearn.metrics.Scorer` as created by
            :meth:`autosklearn.metrics.make_scorer`. These are the `Built-in
            Metrics`_.

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
        # But not if we use share_mode.
        return super(AutoSklearnRegressor, self).fit(X=X, y=y, metric=metric,
                                                     feat_type=feat_type,
                                                     dataset_name=dataset_name)

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.

        """
        return super(AutoSklearnRegressor, self).predict(
            X, batch_size=batch_size, n_jobs=n_jobs)


class AutoMLClassifier(AutoMLDecorator):

    def __init__(self, automl):
        self._classes = []
        self._n_classes = []
        self._n_outputs = 0

        super(AutoMLClassifier, self).__init__(automl)

    def fit(self, X, y,
            metric=None,
            loss=None,
            feat_type=None,
            dataset_name=None):
        X = sklearn.utils.check_array(X, accept_sparse="csr",
                                      force_all_finite=False)
        y = sklearn.utils.check_array(y, ensure_2d=False)

        if scipy.sparse.issparse(X):
            X.sort_indices()

        y_task = type_of_target(y)
        task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                        'multiclass': MULTICLASS_CLASSIFICATION,
                        'binary': BINARY_CLASSIFICATION}

        task = task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)

        if metric is None:
            if task == MULTILABEL_CLASSIFICATION:
                metric = f1_macro
            else:
                metric = accuracy

        y = self._process_target_classes(y)

        return self._automl.fit(X, y, task, metric, feat_type, dataset_name)

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        self._process_target_classes(y)
        return self._automl.fit_ensemble(y, task, metric, precision, dataset_name,
                                         ensemble_nbest, ensemble_size)

    def _process_target_classes(self, y):
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples,), for example using ravel().",
                          sklearn.utils.DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self._n_outputs = y.shape[1]

        y = np.copy(y)

        self._classes = []
        self._n_classes = []

        for k in range(self._n_outputs):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self._classes.append(classes_k)
            self._n_classes.append(classes_k.shape[0])

        self._n_classes = np.array(self._n_classes, dtype=np.int)

        if y.shape[1] == 1:
            y = y.flatten()

        return y

    def predict(self, X, batch_size=None, n_jobs=1):
        predicted_probabilities = self._automl.predict(
            X, batch_size=batch_size, n_jobs=n_jobs)

        if self._n_outputs == 1:
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
            predicted_classes = self._classes[0].take(predicted_indexes)

            return predicted_classes
        else:
            predicted_indices = (predicted_probabilities > 0.5).astype(int)
            n_samples = predicted_probabilities.shape[0]
            predicted_classes = np.zeros((n_samples, self._n_outputs))

            for k in range(self._n_outputs):
                output_predicted_indexes = predicted_indices[:, k].reshape(-1)
                predicted_classes[:, k] = self._classes[k].take(output_predicted_indexes)

            return predicted_classes

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self._automl.predict(X, batch_size=batch_size, n_jobs=n_jobs)


class AutoMLRegressor(AutoMLDecorator):

    def fit(self, X, y,
            metric=None,
            feat_type=None,
            dataset_name=None,
            ):
        if metric is None:
            metric = r2
        return self._automl.fit(X=X, y=y, task=REGRESSION, metric=metric,
                                feat_type=feat_type, dataset_name=dataset_name)
