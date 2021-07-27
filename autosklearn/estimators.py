# -*- encoding: utf-8 -*-
from typing import Optional, Dict, List, Tuple, Union, Iterable, ClassVar
from typing_extensions import Literal

from ConfigSpace.configuration_space import Configuration
import dask.distributed
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target
from smac.runhistory.runhistory import RunInfo, RunValue

from autosklearn.data.validation import (
    SUPPORTED_FEAT_TYPES,
    SUPPORTED_TARGET_TYPES,
)
from autosklearn.pipeline.base import BasePipeline
from autosklearn.automl import AutoMLClassifier, AutoMLRegressor, AutoML
from autosklearn.metrics import Scorer
from autosklearn.util.backend import create



class AutoSklearnEstimator(BaseEstimator):
    
    # Constants used by `def leaderboard` for columns and their sort order
    _leaderboard_columns: ClassVar[Dict[str, List[str]]] = {
        "all": [
            "model_id", "rank", "ensemble_weight", "type", "cost", "duration",
            "config_id", "train_loss", "seed", "start_time", "end_time",
            "budget", "status", "data_preprocessors", "feature_preprocessors",
            "balancing_strategy", "config_origin"
        ],
        "simple": [
            "model_id", "rank", "ensemble_weight", "type", "cost", "duration"
        ]
    }

    def __init__(
        self,
        time_left_for_this_task=3600,
        per_run_time_limit=None,
        initial_configurations_via_metalearning=25,
        ensemble_size: int = 50,
        ensemble_nbest=50,
        max_models_on_disc=50,
        seed=1,
        memory_limit=3072,
        include_estimators=None,
        exclude_estimators=None,
        include_preprocessors=None,
        exclude_preprocessors=None,
        resampling_strategy='holdout',
        resampling_strategy_arguments=None,
        tmp_folder=None,
        delete_tmp_folder_after_terminate=True,
        n_jobs: Optional[int] = None,
        dask_client: Optional[dask.distributed.Client] = None,
        disable_evaluator_output=False,
        get_smac_object_callback=None,
        smac_scenario_args=None,
        logging_config=None,
        metadata_directory=None,
        metric=None,
        scoring_functions: Optional[List[Scorer]] = None,
        load_models: bool = True,
        get_trials_callback=None
    ):
        """
        Parameters
        ----------
        time_left_for_this_task : int, optional (default=3600)
            Time limit in seconds for the search of appropriate
            models. By increasing this value, *auto-sklearn* has a higher
            chance of finding better models.

        per_run_time_limit : int, optional (default=1/10 of time_left_for_this_task)
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
            ensemble.

        max_models_on_disc: int, optional (default=50),
            Defines the maximum number of models that are kept in the disc.
            The additional number of models are permanently deleted. Due to the
            nature of this variable, it sets the upper limit on how many models
            can be used for an ensemble.
            It must be an integer greater or equal than 1.
            If set to None, all models are kept on the disc.

        seed : int, optional (default=1)
            Used to seed SMAC. Will determine the output file names.

        memory_limit : int, optional (3072)
            Memory limit in MB for the machine learning algorithm.
            `auto-sklearn` will stop fitting the machine learning algorithm if
            it tries to allocate more than `memory_limit` MB.
            If None is provided, no memory limit is set.
            In case of multi-processing, `memory_limit` will be per job.
            This memory limit also applies to the ensemble creation process.

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
            * 'cv-iterative-fit': crossvalidation, calls iterative fit where possible
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
            * 'cv-iterative-fit': {'folds': int}
            * 'partial-cv': {'folds': int, 'shuffle': bool}
            * BaseCrossValidator or _RepeatedSplits or BaseShuffleSplit object: all arguments
                required by chosen class as specified in scikit-learn documentation.
                If arguments are not provided, scikit-learn defaults are used.
                If no defaults are available, an exception is raised.
                Refer to the 'n_splits' argument as 'folds'.

        tmp_folder : string, optional (None)
            folder to store configuration output and log files, if ``None``
            automatically use ``/tmp/autosklearn_tmp_$pid_$random_number``

        delete_tmp_folder_after_terminate: string, optional (True)
            remove tmp_folder, when finished. If tmp_folder is None
            tmp_dir will always be deleted

        n_jobs : int, optional, experimental
            The number of jobs to run in parallel for ``fit()``. ``-1`` means
            using all processors. By default, Auto-sklearn uses a single core
            for fitting the machine learning model and a single core for fitting
            an ensemble. Ensemble building is not affected by ``n_jobs`` but
            can be controlled by the number of models in the ensemble. In
            contrast to most scikit-learn models, ``n_jobs`` given in the
            constructor is not applied to the ``predict()`` method. If
            ``dask_client`` is None, a new dask client is created.

        dask_client : dask.distributed.Client, optional
            User-created dask client, can be used to start a dask cluster and then
            attach auto-sklearn to it.

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
            `SMAC documentation <https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario>`_
            for a list of available arguments.

        get_smac_object_callback : callable
            Callback function to create an object of class
            `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html>`_.
            The function must accept the arguments ``scenario_dict``,
            ``instances``, ``num_params``, ``runhistory``, ``seed`` and ``ta``.
            This is an advanced feature. Use only if you are familiar with
            `SMAC <https://automl.github.io/SMAC3/master/index.html>`_.

        logging_config : dict, optional (None)
            dictionary object specifying the logger configuration. If None,
            the default logging.yaml file is used, which can be found in
            the directory ``util/logging.yaml`` relative to the installation.

        metadata_directory : str, optional (None)
            path to the metadata directory. If None, the default directory
            (autosklearn.metalearning.files) is used.

        metric : Scorer, optional (None)
            An instance of :class:`autosklearn.metrics.Scorer` as created by
            :meth:`autosklearn.metrics.make_scorer`. These are the `Built-in
            Metrics`_.
            If None is provided, a default metric is selected depending on the task.

        scoring_functions : List[Scorer], optional (None)
            List of scorers which will be calculated for each pipeline and results will be
            available via ``cv_results``

        load_models : bool, optional (True)
            Whether to load the models after fitting Auto-sklearn.
           
        get_trials_callback: callable
            Callback function to create an object of subclass defined in module
            `smac.callbacks <https://automl.github.io/SMAC3/master/apidoc/smac.callbacks.html>`_.
            This is an advanced feature. Use only if you are familiar with
            `SMAC <https://automl.github.io/SMAC3/master/index.html>`_.

        Attributes
        ----------

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            Not all keys returned by scikit-learn are supported yet.

        """  # noqa (links are too long)
        # Raise error if the given total time budget is less than 60 seconds.
        if time_left_for_this_task < 30:
            raise ValueError("Time left for this task must be at least "
                             "30 seconds. ")
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.seed = seed
        self.memory_limit = memory_limit
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_preprocessors = include_preprocessors
        self.exclude_preprocessors = exclude_preprocessors
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_arguments = resampling_strategy_arguments
        self.tmp_folder = tmp_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.n_jobs = n_jobs
        self.dask_client = dask_client
        self.disable_evaluator_output = disable_evaluator_output
        self.get_smac_object_callback = get_smac_object_callback
        self.smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config
        self.metadata_directory = metadata_directory
        self.metric = metric
        self.scoring_functions = scoring_functions
        self.load_models = load_models
        self.get_trials_callback = get_trials_callback

        self.automl_ = None  # type: Optional[AutoML]

        # Handle the number of jobs and the time for them
        self._n_jobs = None
        if self.n_jobs is None or self.n_jobs == 1:
            self._n_jobs = 1
        elif self.n_jobs == -1:
            self._n_jobs = joblib.cpu_count()
        else:
            self._n_jobs = self.n_jobs

        super().__init__()

    def __getstate__(self):
        # Cannot serialize a client!
        self.dask_client = None
        return self.__dict__

    def build_automl(self):

        backend = create(
            temporary_directory=self.tmp_folder,
            delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
            )

        automl = self._get_automl_class()(
            backend=backend,
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            initial_configurations_via_metalearning=self.initial_configurations_via_metalearning,
            ensemble_size=self.ensemble_size,
            ensemble_nbest=self.ensemble_nbest,
            max_models_on_disc=self.max_models_on_disc,
            seed=self.seed,
            memory_limit=self.memory_limit,
            include_estimators=self.include_estimators,
            exclude_estimators=self.exclude_estimators,
            include_preprocessors=self.include_preprocessors,
            exclude_preprocessors=self.exclude_preprocessors,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_arguments=self.resampling_strategy_arguments,
            n_jobs=self._n_jobs,
            dask_client=self.dask_client,
            get_smac_object_callback=self.get_smac_object_callback,
            disable_evaluator_output=self.disable_evaluator_output,
            smac_scenario_args=self.smac_scenario_args,
            logging_config=self.logging_config,
            metadata_directory=self.metadata_directory,
            metric=self.metric,
            scoring_functions=self.scoring_functions,
            get_trials_callback=self.get_trials_callback
        )

        return automl

    def fit(self, **kwargs):

        # Automatically set the cutoff time per task
        if self.per_run_time_limit is None:
            self.per_run_time_limit = self._n_jobs * self.time_left_for_this_task // 10

        if self.automl_ is None:
            self.automl_ = self.build_automl()
        self.automl_.fit(load_models=self.load_models, **kwargs)

        return self

    def fit_pipeline(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        config: Union[Configuration,  Dict[str, Union[str, float, int]]],
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
        feat_type: Optional[List[str]] = None,
        *args,
        **kwargs: Dict,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        """ Fits and individual pipeline configuration and returns
        the result to the user.

        The Estimator constraints are honored, for example the resampling
        strategy, or memory constraints, unless directly provided to the method.
        By default, this method supports the same signature as fit(), and any extra
        arguments are redirected to the TAE evaluation function, which allows for
        further customization while building a pipeline.

        Any additional argument provided is directly passed to the worker exercising the run.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            The features used for training
        y: array-like
            The labels used for training
        X_test: Optionalarray-like, shape = (n_samples, n_features)
            If provided, the testing performance will be tracked on this features.
        y_test: array-like
            If provided, the testing performance will be tracked on this labels
        config: Union[Configuration,  Dict[str, Union[str, float, int]]]
            A configuration object used to define the pipeline steps.
            If a dictionary is passed, a configuration is created based on this dictionary.
        dataset_name: Optional[str]
            Name that will be used to tag the Auto-Sklearn run and identify the
            Auto-Sklearn run
        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        Returns
        -------
        pipeline: Optional[BasePipeline]
            The fitted pipeline. In case of failure while fitting the pipeline,
            a None is returned.
        run_info: RunInFo
            A named tuple that contains the configuration launched
        run_value: RunValue
            A named tuple that contains the result of the run
        """
        if self.automl_ is None:
            self.automl_ = self.build_automl()
        return self.automl_.fit_pipeline(X=X, y=y,
                                         dataset_name=dataset_name,
                                         config=config,
                                         feat_type=feat_type,
                                         X_test=X_test, y_test=y_test,
                                         *args, **kwargs)

    def fit_ensemble(self, y, task=None, precision=32,
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
        if self.automl_ is None:
            # Build a dummy automl object to call fit_ensemble
            # The ensemble size is honored in the .automl_.fit_ensemble
            # call
            self.automl_ = self.build_automl()
        self.automl_.fit_ensemble(
            y=y,
            task=task,
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
        self.automl_.refit(X, y)
        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        return self.automl_.predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self.automl_.predict_proba(
             X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, X, y):
        return self.automl_.score(X, y)

    def show_models(self):
        """Return a representation of the final ensemble found by auto-sklearn.

        Returns
        -------
        str

        """
        return self.automl_.show_models()

    def get_models_with_weights(self):
        """Return a list of the final ensemble found by auto-sklearn.

        Returns
        -------
        [(weight_1, model_1), ..., (weight_n, model_n)]

        """
        return self.automl_.get_models_with_weights()

    @property
    def cv_results_(self):
        return self.automl_.cv_results_

    @property
    def trajectory_(self):
        return self.automl_.trajectory_

    @property
    def fANOVA_input_(self):
        return self.automl_.fANOVA_input_

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
        return self.automl_.sprint_statistics()

    def leaderboard(
        self,
        detailed: bool = False,
        ensemble_only: bool = True,
        top_k: Union[int, Literal['all']] = 'all',
        sort_by: str = 'cost',
        sort_order: Literal['auto', 'ascending', 'descending'] = 'auto',
        include: Optional[Union[str, Iterable[str]]] = None
    ) -> pd.DataFrame:
        """ Returns a pandas table of results for all evaluated models.

        Gives an overview of all models trained during the search process along
        with various statistics about their training.

        The availble statistics are:

        **Simple**:

        * ``"model_id"`` - The id given to a model by ``autosklearn``.
        * ``"rank"`` - The rank of the model based on it's ``"cost"``.
        * ``"ensemble_weight"`` - The weight given to the model in the ensemble.
        * ``"type"`` - The type of classifier/regressor used.
        * ``"cost"`` - The loss of the model on the validation set.
        * ``"duration"`` - Length of time the model was optimized for.

        **Detailed**:
        The detailed view includes all of the simple statistics along with the
        following.

        * ``"config_id"`` - The id used by SMAC for optimization.
        * ``"budget"`` - How much time was allocated to this model.
        * ``"status"`` - The return status of training the model with SMAC.
        * ``"train_loss"`` - The loss of the model on the training set.
        * ``"balancing_strategy"`` - The balancing strategy used for data preprocessing.
        * ``"start_time"`` - Time the model began being optimized
        * ``"end_time"`` - Time the model ended being optimized
        * ``"data_preprocessors"`` - The preprocessors used on the data
        * ``"feature_preprocessors"`` - The preprocessors for features types

        Parameters
        ----------
        detailed: bool = False
            Whether to give detailed information or just a simple overview.

        ensemble_only: bool = True
            Whether to view only models included in the ensemble or all models
            trained.

        top_k: int or "all" = "all"
            How many models to display.

        sort_by: str = 'cost'
            What column to sort by. If that column is not present, the
            sorting defaults to the ``"model_id"`` index column.

        sort_order: "auto" or "ascending" or "descending" = "auto"
            Which sort order to apply to the ``sort_by`` column. If left
            as ``"auto"``, it will sort by a sensible default where "better" is
            on top, otherwise defaulting to the pandas default for
            `DataFrame.sort_values`_ if there is no obvious "better".

            .. _DataFrame.sort_values: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

        include: Optional[str or Iterable[str]]
            Items to include, other items not specified will be excluded.
            The exception is the ``"model_id"`` index column which is always included.

            If left as ``None``, it will resort back to using the ``detailed``
            param to decide the columns to include.

        Returns
        -------
        pd.DataFrame
            A dataframe of statistics for the models, ordered by ``sort_by``.

        """  # noqa (links are too long)
        # TODO validate that `self` is fitted. This is required for
        #      self.ensemble_ to get the identifiers of models it will generate
        #      weights for.
        column_types = {
            'all' : AutoSklearnEstimator._leaderboard_columns['all'],
            'simple': AutoSklearnEstimator._leaderboard_columns['simple'],
            'detailed': AutoSklearnEstimator._leaderboard_columns['all']
        }

        # Validation of top_k
        if (
            not (isinstance(top_k, str) or isinstance(top_k, int))
            or (isinstance(top_k, str) and top_k != 'all')
            or (isinstance(top_k, int) and top_k <= 0)
        ):
            raise ValueError(f"top_k={top_k} must be a positive integer or pass"
                             " `top_k`='all' to view results for all models")

        # Validate columns to include
        if isinstance(include, str):
            include = [include]

        if include is not None:
            columns = [*include]

            # 'model_id' should always be present as it is the unique index
            # used for pandas
            if 'model_id' not in columns:
                columns.append('model_id')

            invalid_include_items = set(columns) - set(column_types['all'])
            if len(invalid_include_items) != 0:
                raise ValueError(f"Values {invalid_include_items} are not known"
                                 f" columns to include, must be contained in "
                                 f"{column_types['all']}")
        elif detailed:
            columns = column_types['all']
        else:
            columns = column_types['simple']

        # Validation of sorting
        if sort_by not in column_types['all']:
            raise ValueError(f"sort_by='{sort_by}' must be one of included "
                             f"columns {set(column_types['all'])}")

        valid_sort_orders = ['auto', 'ascending', 'descending']
        if not (isinstance(sort_order, str) and sort_order in valid_sort_orders):
            raise ValueError(f"`sort_order` = {sort_order} must be a str in "
                             f"{valid_sort_orders}")

        # To get all the models that were optmized, we collect what we can from
        # runhistory first.
        def has_key(rv, key):
            return rv.additional_info and key in rv.additional_info

        model_runs = {
            rval.additional_info['num_run']: {
                'model_id': rval.additional_info['num_run'],
                'seed': rkey.seed,
                'budget': rkey.budget,
                'duration': rval.time,
                'config_id': rkey.config_id,
                'start_time': rval.starttime,
                'end_time': rval.endtime,
                'status': str(rval.status),
                'cost': rval.cost,
                'train_loss': rval.additional_info['train_loss']
                if has_key(rval, 'train_loss') else None,
                'config_origin': rval.additional_info['configuration_origin']
                if has_key(rval, 'configuration_origin') else None
            }
            for rkey, rval in self.automl_.runhistory_.data.items()
            if has_key(rval, 'num_run')
        }

        # Next we get some info about the model itself
        model_class_strings = {
            AutoMLClassifier: 'classifier',
            AutoMLRegressor: 'regressor'
        }
        model_type = model_class_strings.get(self._get_automl_class(), None)
        if model_type is None:
            raise RuntimeError(f"Unknown `automl_class` {self._get_automl_class()}")

        # A dict mapping model ids to their configurations
        configurations = self.automl_.runhistory_.ids_config

        for model_id, run_info in model_runs.items():
            config_id = run_info['config_id']
            run_config = configurations[config_id]._values

            run_info.update({
                'balancing_strategy': run_config.get('balancing:strategy', None),
                'type': run_config[f'{model_type}:__choice__'],
                'data_preprocessors': [
                    value for key, value in run_config.items()
                    if 'data_preprocessing' in key and '__choice__' in key
                ],
                'feature_preprocessors': [
                    value for key, value in run_config.items()
                    if 'feature_preprocessor' in key and '__choice__' in key
                ]
            })

        # Get the models ensemble weight if it has one
        # TODO both implementing classes of AbstractEnsemble have a property
        #      `identifiers_` and `weights_`, might be good to put it as an
        #       abstract property
        # TODO `ensemble_.identifiers_` and `ensemble_.weights_` are loosely
        #      tied together by ordering, might be better to store as tuple
        for i, weight in enumerate(self.automl_.ensemble_.weights_):
            (_, model_id, _) = self.automl_.ensemble_.identifiers_[i]
            model_runs[model_id]['ensemble_weight'] = weight

        # Filter out non-ensemble members if needed, else fill in a default
        # value of 0 if it's missing
        if ensemble_only:
            model_runs = {
                model_id: info
                for model_id, info in model_runs.items()
                if ('ensemble_weight' in info and info['ensemble_weight'] > 0)
            }
        else:
            for model_id, info in model_runs.items():
                if 'ensemble_weight' not in info:
                    info['ensemble_weight'] = 0

        # `rank` relies on `cost` so we include `cost`
        # We drop it later if it's not requested
        if 'rank' in columns and 'cost' not in columns:
            columns = [*columns, 'cost']

        # Finally, convert into a tabular format by converting the dict into
        # column wise orientation.
        dataframe = pd.DataFrame({
            col: [run_info[col] for run_info in model_runs.values()]
            for col in columns if col != 'rank'
        })

        # Give it an index, even if not in the `include`
        dataframe.set_index('model_id', inplace=True)

        # Add the `rank` column if needed, dropping `cost` if it's not
        # requested by the user
        if 'rank' in columns:
            dataframe.sort_values(by='cost', ascending=False, inplace=True)
            dataframe.insert(column='rank',
                             value=range(1, len(dataframe) + 1),
                             loc=list(columns).index('rank'))

            if 'cost' not in columns:
                dataframe.drop('cost', inplace=True)

        # Decide on the sort order depending on what it gets sorted by
        descending_columns = ['ensemble_weight', 'duration']
        if sort_order == 'auto':
            ascending_param = False if sort_by in descending_columns else True
        else:
            ascending_param = False if sort_order == 'descending' else True

        # Sort by the given column name, defaulting to 'model_id' if not present
        if sort_by not in dataframe.columns:
            self.automl_._logger.warning(f"sort_by = '{sort_by}' was not present"
                                         ", defaulting to sort on the index "
                                         "'model_id'")
            sort_by = 'model_id'

        dataframe.sort_values(by=sort_by,
                              ascending=ascending_param,
                              inplace=True)

        # Lastly, just grab the top_k
        if top_k == 'all' or top_k >= len(dataframe):
            top_k = len(dataframe)

        dataframe = dataframe.head(top_k)

        return dataframe

    def _get_automl_class(self):
        raise NotImplementedError()

    def get_configuration_space(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
        dataset_name: Optional[str] = None,
        feat_type: Optional[List[str]] = None,
    ):
        """
        Returns the Configuration Space object, from which Auto-Sklearn
        will sample configurations and build pipelines.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Array with the training features, used to get characteristics like
            data sparsity
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Array with the problem labels
        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Array with features used for performance estimation
        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Array with the problem labels for the testing split
        dataset_name: Optional[str]
            A string to tag the Auto-Sklearn run
        """
        if self.automl_ is None:
            self.automl_ = self.build_automl()
        return self.automl_.fit(
            X, y,
            X_test=X_test, y_test=y_test,
            dataset_name=dataset_name,
            feat_type=feat_type,
            only_return_configuration_space=True,
        ) if self.automl_.configuration_space is None else self.automl_.configuration_space



class AutoSklearnClassifier(AutoSklearnEstimator, ClassifierMixin):
    """
    This class implements the classification task.

    """

    def fit(self, X, y,
            X_test=None,
            y_test=None,
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

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

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
        supported_types = ['binary', 'multiclass', 'multilabel-indicator']
        if target_type not in supported_types:
            raise ValueError("Classification with data of type {} is "
                             "not supported. Supported types are {}. "
                             "You can find more information about scikit-learn "
                             "data types in: "
                             "https://scikit-learn.org/stable/modules/multiclass.html"
                             "".format(
                                    target_type,
                                    supported_types
                                )
                             )

        # remember target type for using in predict_proba later.
        self.target_type = target_type

        super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        # After fit, a classifier is expected to define classes_
        # A list of class labels known to the classifier, mapping each label
        # to a numerical index used in the model representation our output.
        self.classes_ = self.automl_.InputValidator.target_validator.classes_

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


class AutoSklearnRegressor(AutoSklearnEstimator, RegressorMixin):
    """
    This class implements the regression task.

    """

    def fit(self, X, y,
            X_test=None,
            y_test=None,
            feat_type=None,
            dataset_name=None):
        """Fit *Auto-sklearn* to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            The regression target.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples] or [n_samples, n_targets]
            The regression target. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

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
        # types are: continuous, continuous-multioutput, and the special cases:
        # multiclass : because [3.0, 1.0, 5.0] is considered as multiclass
        # binary: because [1.0, 0.0] is considered multiclass
        target_type = type_of_target(y)
        supported_types = ['continuous', 'binary', 'multiclass', 'continuous-multioutput']
        if target_type not in supported_types:
            raise ValueError("Regression with data of type {} is "
                             "not supported. Supported types are {}. "
                             "You can find more information about scikit-learn "
                             "data types in: "
                             "https://scikit-learn.org/stable/modules/multiclass.html"
                             "".format(
                                    target_type,
                                    supported_types
                                )
                             )

        # Fit is supposed to be idempotent!
        # But not if we use share_mode.
        super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
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
