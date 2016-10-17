# -*- encoding: utf-8 -*-

import numpy as np
import six
import warnings

import autosklearn.automl
from autosklearn.constants import *
from autosklearn.util.backend import create
from sklearn.base import BaseEstimator
import sklearn.utils
import scipy.sparse


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
        """Build the ensemble.

        This method only needs to be called in the parallel mode.

        Returns
        -------
        self
        """
        return self._automl.fit_ensemble(y, task, metric, precision,
                                         dataset_name, ensemble_nbest,
                                         ensemble_size)

    def predict(self, X):
        return self._automl.predict(X)

    def score(self, X, y):
        return self._automl.score(X, y)

    def show_models(self):
        """Return a representation of the final ensemble found by auto-sklearn

        Returns
        -------
        str
        """
        return self._automl.show_models()

    @property
    def grid_scores_(self):
        return self._automl.grid_scores_

    @property
    def cv_results_(self):
        return self._automl.cv_results_

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
            Number of models added to the ensemble built by `Ensemble
            selection from libraries of models. Models are drawn with
            replacement.

        ensemble_nbest : int, optional (default=50)
            Only consider the ``ensemble_nbest`` models when building an
            ensemble. Implements `Model Library Pruning` from `Getting the
            most out of ensemble selection`.

        seed : int, optional (default=1)

        ml_memory_limit : int, optional (3000)
            Memory limit in MB for the machine learning algorithm.
            `auto-sklearn` will stop fitting the machine learning algorithm if
            it tries to allocate more than `ml_memory_limit` MB.

        include_estimators : dict, optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators to use

        include_preprocessors : dict, optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors to use

        resampling_strategy : string, optional ('holdout')
            how to to handle overfitting, might need 'resampling_strategy_arguments'

            * 'holdout': 66:33 (train:test) split
            * 'holdout-iterative-fit':  66:33 (train:test) split, calls iterative
              fit where possible
            * 'cv': crossvalidation, requires 'folds'
            * 'nested-cv': crossvalidation, requires 'outer-folds, 'inner-folds'

        resampling_strategy_arguments : dict, optional if 'holdout' (None)
            Additional arguments for resampling_strategy
            * 'holdout': None
            * 'holdout-iterative-fit':  None
            * 'cv': {'folds': int}
            * 'nested-cv': {'outer_folds': int, 'inner_folds'

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

        Attributes
        ----------
        grid_scores\_ : list of named tuples
            Contains scores for all parameter combinations in param_grid.
            Each entry corresponds to one parameter setting.
            Each named tuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            This attribute is a backward port to already support the advanced
            output of scikit-learn 0.18. Not all keys returned by scikit-learn
            are supported yet.

        """
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.ml_memory_limit = ml_memory_limit
        self.include_estimators = include_estimators
        self.include_preprocessors = include_preprocessors
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_arguments = resampling_strategy_arguments
        self.tmp_folder = tmp_folder
        self.output_folder = output_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        self.shared_mode = shared_mode
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
            include_preprocessors=self.include_preprocessors,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_arguments=self.resampling_strategy_arguments,
            delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=
            self.delete_output_folder_after_terminate,
            shared_mode=self.shared_mode)

        return automl

    def fit(self, *args, **kwargs):
        self._automl = self.build_automl()
        super(AutoSklearnEstimator, self).fit(*args, **kwargs)

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
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
            metric='acc_metric',
            feat_type=None,
            dataset_name=None):
        """Fit *auto-sklearn* to given training set (X, y).

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
        return super(AutoSklearnClassifier, self).fit(X, y, metric, feat_type, dataset_name)

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
        return self._automl.predict_proba(X)


class AutoSklearnRegressor(AutoSklearnEstimator):
    """
    This class implements the regression task.
    """

    def build_automl(self):
        automl = super(AutoSklearnRegressor, self).build_automl()
        return AutoMLRegressor(automl)

    def fit(self, X, y,
            metric='r2_metric',
            feat_type=None,
            dataset_name=None):
        """Fit *autosklearn* to given training set (X, y).

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The regression target.

        metric : str, optional (default='r2_metric')
            The metric to optimize for. Can be one of: ['r2_metric',
            'a_metric']. A description of the metrics can be found in
            `the paper describing the AutoML Challenge
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
        # But not if we use share_mode.
        return super(AutoSklearnRegressor, self).fit(X, y, metric, feat_type, dataset_name)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.

        """
        return super(AutoSklearnRegressor, self).predict(X)


class AutoMLClassifier(AutoMLDecorator):

    def __init__(self, automl):
        self._classes = []
        self._n_classes = []
        self._n_outputs = 0

        super(AutoMLClassifier, self).__init__(automl)

    def fit(self, X, y,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None,
            ):
        # From sklearn.tree.DecisionTreeClassifier
        X = sklearn.utils.check_array(X, accept_sparse="csr",
                                      force_all_finite=False)
        if scipy.sparse.issparse(X):
            X.sort_indices()

        y = self._process_target_classes(y)

        if self._n_outputs > 1:
            task = MULTILABEL_CLASSIFICATION
        else:
            if len(self._classes[0]) == 2:
                task = BINARY_CLASSIFICATION
            else:
                task = MULTICLASS_CLASSIFICATION

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

        for k in six.moves.range(self._n_outputs):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self._classes.append(classes_k)
            self._n_classes.append(classes_k.shape[0])

        self._n_classes = np.array(self._n_classes, dtype=np.int)

        # TODO: fix metafeatures calculation to allow this!
        if y.shape[1] == 1:
            y = y.flatten()

        return y


    def predict(self, X):
        predicted_probabilities = self._automl.predict(X)
        if self._n_outputs == 1:
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
            predicted_classes = self._classes[0].take(predicted_indexes)

            return predicted_classes
        else:
            argmax_v = np.vectorize(np.argmax, otypes=[int])
            predicted_indexes = argmax_v(predicted_probabilities)
            #predicted_indexes = np.argmax(predicted_probabilities, axis=1)
            n_samples = predicted_probabilities.shape[0]
            predicted_classes = np.zeros((n_samples, self._n_outputs), dtype=object)

            for k in six.moves.range(self._n_outputs):
                output_predicted_indexes = predicted_indexes[:, k].reshape(-1)
                predicted_classes[:, k] = self._classes[k].take(output_predicted_indexes)

            return predicted_classes

    def predict_proba(self, X):
        return self._automl.predict(X)


class AutoMLRegressor(AutoMLDecorator):

    def fit(self, X, y,
            metric='r2_metric',
            feat_type=None,
            dataset_name=None,
            ):
        return self._automl.fit(X, y, REGRESSION, metric, feat_type, dataset_name)
