# -*- encoding: utf-8 -*-
import copy
import multiprocessing
from typing import Optional, List, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target

from autosklearn.automl import AutoMLClassifier, AutoMLRegressor, BaseAutoML
from autosklearn.util.backend import create, get_randomized_directory_names


def _fit_automl(automl, kwargs, load_models):
    return automl.fit(load_models=load_models, **kwargs)


class AutoSklearnEstimator(BaseEstimator):

    def __init__(
        self,
        time_left_for_this_task=3600,
        per_run_time_limit=360,
        initial_configurations_via_metalearning=25,
        ensemble_size: int = 50,
        ensemble_nbest=50,
        ensemble_memory_limit=1024,
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
        n_jobs: Optional[int] = None,
        disable_evaluator_output=False,
        get_smac_object_callback=None,
        smac_scenario_args=None,
        logging_config=None,
        metadata_directory=None,
    ):
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

        ensemble_memory_limit : int, optional (1024)
            Memory limit in MB for the ensemble building process.
            `auto-sklearn` will reduce the number of considered models
            (``ensemble_nbest``) if the memory limit is reached.

        seed : int, optional (default=1)
            Used to seed SMAC. Will determine the output file names.

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

        resampling_strategy : string or object, optional ('holdout')
            how to to handle overfitting, might need 'resampling_strategy_arguments'

            * 'holdout': 67:33 (train:test) split
            * 'holdout-iterative-fit':  67:33 (train:test) split, calls iterative
              fit where possible
            * 'cv': crossvalidation, requires 'folds'
            * 'partial-cv': crossvalidation with intensification, requires
              'folds'
            * BaseCrossValidator object: any BaseCrossValidator class found
                                        in scikit-learn model_selection module
            * _RepeatedSplits object: any _RepeatedSplits class found
                                      in scikit-learn model_selection module
            * BaseShuffleSplit object: any BaseShuffleSplit class found
                                      in scikit-learn model_selection module

        resampling_strategy_arguments : dict, optional if 'holdout' (train_size default=0.67)
            Additional arguments for resampling_strategy:

            * ``train_size`` should be between 0.0 and 1.0 and represent the
              proportion of the dataset to include in the train split.
            * ``shuffle`` determines whether the data is shuffled prior to
              splitting it into train and validation.

            Available arguments:

            * 'holdout': {'train_size': float}
            * 'holdout-iterative-fit':  {'train_size': float}
            * 'cv': {'folds': int}
            * 'partial-cv': {'folds': int, 'shuffle': bool}
            * BaseCrossValidator or _RepeatedSplits or BaseShuffleSplit object: all arguments
                required by chosen class as specified in scikit-learn documentation.
                If arguments are not provided, scikit-learn defaults are used.
                If no defaults are available, an exception is raised.
                Refer to the 'n_splits' argument as 'folds'.

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

        shared_mode : bool, optional (False)
            Run smac in shared-model-node. This only works if arguments
            ``tmp_folder`` and ``output_folder`` are given and both
            ``delete_tmp_folder_after_terminate`` and
            ``delete_output_folder_after_terminate`` are set to False. Cannot
            be used together with ``n_jobs``.

        n_jobs : int, optional, experimental
            The number of jobs to run in parallel for ``fit()``. Cannot be
            used together with ``shared_mode``. ``-1`` means using all
            processors. By default, Auto-sklearn uses a single core for
            fitting the machine learning model and a single core for fitting
            an ensemble. Ensemble building is not affected by ``n_jobs`` but
            can be controlled by the number of models in the ensemble. In
            contrast to most scikit-learn models, ``n_jobs`` given in the
            constructor is not applied to the ``predict()`` method.

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

        smac_scenario_args : dict, optional (None)
            Additional arguments inserted into the scenario of SMAC. See the
            `SMAC documentation <https://automl.github.io/SMAC3/stable/options.html?highlight=scenario#scenario>`_
            for a list of available arguments.

        get_smac_object_callback : callable
            Callback function to create an object of class
            `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/stable/apidoc/smac.optimizer.smbo.html>`_.
            The function must accept the arguments ``scenario_dict``,
            ``instances``, ``num_params``, ``runhistory``, ``seed`` and ``ta``.
            This is an advanced feature. Use only if you are familiar with
            `SMAC <https://automl.github.io/SMAC3/stable/index.html>`_.

        logging_config : dict, optional (None)
            dictionary object specifying the logger configuration. If None,
            the default logging.yaml file is used, which can be found in
            the directory ``util/logging.yaml`` relative to the installation.

        metadata_directory : str, optional (None)
            path to the metadata directory. If None, the default directory
            (autosklearn.metalearning.files) is used.

        Attributes
        ----------

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            Not all keys returned by scikit-learn are supported yet.

        """
        # Raise error if the given total time budget is less than 60 seconds.
        if time_left_for_this_task < 30:
            raise ValueError("Time left for this task must be at least "
                             "30 seconds. ")
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.ensemble_memory_limit = ensemble_memory_limit
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
        self.n_jobs = n_jobs
        self.disable_evaluator_output = disable_evaluator_output
        self.get_smac_object_callback = get_smac_object_callback
        self.smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config
        self.metadata_directory = metadata_directory

        self._automl = None  # type: Optional[List[BaseAutoML]]
        # n_jobs after conversion to a number (b/c default is None)
        self._n_jobs = None
        super().__init__()

    def build_automl(
        self,
        seed: int,
        shared_mode: bool,
        ensemble_size: int,
        initial_configurations_via_metalearning: int,
        tmp_folder: str,
        output_folder: str,
        smac_scenario_args: Optional[Dict] = None,
    ):

        if shared_mode:
            self.delete_output_folder_after_terminate = False
            self.delete_tmp_folder_after_terminate = False
            if tmp_folder is None:
                raise ValueError("If shared_mode == True tmp_folder must not "
                                 "be None.")
            if output_folder is None:
                raise ValueError("If shared_mode == True output_folder must "
                                 "not be None.")

        backend = create(temporary_directory=tmp_folder,
                         output_directory=output_folder,
                         delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
                         delete_output_folder_after_terminate=self.delete_output_folder_after_terminate,
                         shared_mode=shared_mode)

        if smac_scenario_args is None:
            smac_scenario_args = self.smac_scenario_args

        automl = self._get_automl_class()(
            backend=backend,
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            initial_configurations_via_metalearning=
            initial_configurations_via_metalearning,
            ensemble_size=ensemble_size,
            ensemble_nbest=self.ensemble_nbest,
            ensemble_memory_limit=self.ensemble_memory_limit,
            seed=seed,
            ml_memory_limit=self.ml_memory_limit,
            include_estimators=self.include_estimators,
            exclude_estimators=self.exclude_estimators,
            include_preprocessors=self.include_preprocessors,
            exclude_preprocessors=self.exclude_preprocessors,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_arguments=self.resampling_strategy_arguments,
            shared_mode=shared_mode,
            get_smac_object_callback=self.get_smac_object_callback,
            disable_evaluator_output=self.disable_evaluator_output,
            smac_scenario_args=smac_scenario_args,
            logging_config=self.logging_config,
            metadata_directory=self.metadata_directory,
        )

        return automl

    def fit(self, **kwargs):
        self._automl = []
        if self.shared_mode and self.n_jobs:
            raise ValueError(
                'Cannot enable both shared_model and n_jobs. Please specify '
                'only one of them.'
            )

        if self.n_jobs is None or self.n_jobs == 1:
            self._n_jobs = 1
            shared_mode = self.shared_mode
            seed = self.seed
            automl = self.build_automl(
                seed=seed,
                shared_mode=shared_mode,
                ensemble_size=self.ensemble_size,
                initial_configurations_via_metalearning=(
                    self.initial_configurations_via_metalearning
                ),
                tmp_folder=self.tmp_folder,
                output_folder=self.output_folder,
            )
            self._automl.append(automl)
            self._automl[0].fit(**kwargs)
        else:
            tmp_folder, output_folder = get_randomized_directory_names(
                temporary_directory=self.tmp_folder,
                output_directory=self.output_folder,
            )

            self._n_jobs = self.n_jobs
            shared_mode = True
            seeds = set()
            for i in range(self._n_jobs):
                rs = np.random.RandomState(self.seed + i)
                while True:
                    seed = int(rs.randint(0, 2 ** 32))
                    if seed in seeds:
                        continue
                    else:
                        break

                if i != 0:
                    smac_scenario_args = copy.deepcopy(self.smac_scenario_args)
                    if smac_scenario_args is None:
                        smac_scenario_args = dict()
                    if 'initial_incumbent' not in smac_scenario_args:
                        smac_scenario_args['initial_incumbent'] = 'RANDOM'
                else:
                    smac_scenario_args = self.smac_scenario_args

                automl = self.build_automl(
                    seed=seed,
                    shared_mode=shared_mode,
                    # Start the ensemble process only for the first AutoML
                    # process (the first AutoML will be executed in the
                    # current process, too)
                    ensemble_size=self.ensemble_size if i == 0 else 0,
                    initial_configurations_via_metalearning=(
                        self.initial_configurations_via_metalearning
                        if i == 0
                        else 0
                    ),
                    tmp_folder=tmp_folder,
                    output_folder=output_folder,
                    smac_scenario_args=smac_scenario_args,
                )
                self._automl.append(automl)
            # Start all except for the first instances of Auto-sklearn in a
            # new process!
            processes = []
            for i in range(1, self._n_jobs):
                p = multiprocessing.Process(
                    target=_fit_automl,
                    kwargs=dict(
                        automl=self._automl[i],
                        kwargs=kwargs,
                        load_models=False,
                    ),
                )
                processes.append(p)
                p.start()
            _fit_automl(
                automl=self._automl[0],
                kwargs=kwargs,
                load_models=True,
            )
            for p in processes:
                p.join()

        return self

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        """Fit an ensemble to models trained during an optimization process.

        All parameters are ``None`` by default. If no other value is given,
        the default values which were set in a call to ``fit()`` are used.

        Calling this function is only necessary if ``ensemble_size==0``, for
        example when executing *auto-sklearn* in parallel.

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
            Size of the ensemble built by `Ensemble Selection`.

        Returns
        -------
        self

        """
        if self._automl is None:
            if self.n_jobs is None or self.n_jobs == 1:
                shared_mode = self.shared_mode
            else:
                shared_mode = True
            # Build a dummy automl object to call fit_ensemble
            self._automl = [
                self.build_automl(
                    seed=self.seed,
                    shared_mode=shared_mode,
                    ensemble_size=(
                        ensemble_size
                        if ensemble_size is not None else
                        self.ensemble_size
                    ),
                    initial_configurations_via_metalearning=0,
                    tmp_folder=self.tmp_folder,
                    output_folder=self.output_folder,
                )
            ]
        self._automl[0].fit_ensemble(
            y=y,
            task=task,
            metric=metric,
            precision=precision,
            dataset_name=dataset_name,
            ensemble_nbest=ensemble_nbest,
            ensemble_size=ensemble_size,
        )
        return self

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
        self._automl[0].refit(X, y)
        return self


    def predict(self, X, batch_size=None, n_jobs=1):
        return self._automl[0].predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self._automl[0].predict_proba(
             X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, X, y):
        return self._automl[0].score(X, y)

    def show_models(self):
        """Return a representation of the final ensemble found by auto-sklearn.

        Returns
        -------
        str

        """
        return self._automl[0].show_models()

    def get_models_with_weights(self):
        """Return a list of the final ensemble found by auto-sklearn.

        Returns
        -------
        [(weight_1, model_1), ..., (weight_n, model_n)]

        """
        return self._automl[0].get_models_with_weights()

    @property
    def cv_results_(self):
        return self._automl[0].cv_results_

    @property
    def trajectory_(self):
        if len(self._automl) > 1:
            raise NotImplementedError()
        return self._automl[0].trajectory_

    @property
    def fANOVA_input_(self):
        if len(self._automl) > 1:
            raise NotImplementedError()
        return self._automl[0].fANOVA_input_

    def sprint_statistics(self):
        """Return the following statistics of the training result:

        - dataset name
        - metric used
        - best validation score
        - number of target algorithm runs
        - number of successful target algorithm runs
        - number of crashed target algorithm runs
        - number of target algorithm runs that exceeded the memory limit
        - number of target algorithm runs that exceeded the time limit

        Returns
        -------
        str
        """
        return self._automl[0].sprint_statistics()

    def _get_automl_class(self):
        raise NotImplementedError()

    def get_configuration_space(self, X, y):
        self._automl = self.build_automl()
        return self._automl[0].fit(X, y, only_return_configuration_space=True)


class AutoSklearnClassifier(AutoSklearnEstimator):
    """
    This class implements the classification task.

    """

    def fit(self, X, y,
            X_test=None,
            y_test=None,
            metric=None,
            feat_type=None,
            dataset_name=None):
        """Fit *auto-sklearn* to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Test data target classes. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

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
        # Before running anything else, first check that the
        # type of data is compatible with auto-sklearn. Legal target
        # types are: binary, multiclass, multilabel-indicator.
        target_type = type_of_target(y)
        if target_type in ['multiclass-multioutput',
                           'continuous',
                           'continuous-multioutput',
                           'unknown',
                           ]:
            raise ValueError("classification with data of type %s is"
                             " not supported" % target_type)

        # remember target type for using in predict_proba later.
        self.target_type = target_type

        super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        return self

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
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):

        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        batch_size : int (optional)
            Number of data points to predict for (predicts all points at once
            if ``None``.
        n_jobs : int

        Returns
        -------
        y : array of shape = [n_samples, n_classes] or [n_samples, n_labels]
            The predicted class probabilities.

        """
        pred_proba = super().predict_proba(
            X, batch_size=batch_size, n_jobs=n_jobs)

        # Check if all probabilities sum up to 1.
        # Assert only if target type is not multilabel-indicator.
        if self.target_type not in ['multilabel-indicator']:
            assert(
                np.allclose(
                    np.sum(pred_proba, axis=1),
                    np.ones_like(pred_proba[:, 0]))
            ), "prediction probability does not sum up to 1!"

        # Check that all probability values lie between 0 and 1.
        assert(
            (pred_proba >= 0).all() and (pred_proba <= 1).all()
        ), "found prediction probability value outside of [0, 1]!"

        return pred_proba

    def _get_automl_class(self):
        return AutoMLClassifier


class AutoSklearnRegressor(AutoSklearnEstimator):
    """
    This class implements the regression task.

    """

    def fit(self, X, y,
            X_test=None,
            y_test=None,
            metric=None,
            feat_type=None,
            dataset_name=None):
        """Fit *Auto-sklearn* to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The regression target.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples]
            The regression target. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

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
        # Before running anything else, first check that the
        # type of data is compatible with auto-sklearn. Legal target
        # types are: continuous, binary, multiclass.
        target_type = type_of_target(y)
        if target_type in ['multiclass-multioutput',
                           'multilabel-indicator',
                           'continuous-multioutput',
                           'unknown',
                           ]:
            raise ValueError("regression with data of type %s is not"
                             " supported" % target_type)

        # Fit is supposed to be idempotent!
        # But not if we use share_mode.
        super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        return self

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
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def _get_automl_class(self):
        return AutoMLRegressor
