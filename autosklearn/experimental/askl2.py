import hashlib
import json
import os
import pathlib
import pickle
from typing import Any, Dict, List, Optional, Union

import dask.distributed

from ConfigSpace import Configuration
import numpy as np
import pandas as pd
import sklearn

import autosklearn
from autosklearn.classification import AutoSklearnClassifier
import autosklearn.experimental.selector
from autosklearn.metrics import Scorer

this_directory = pathlib.Path(__file__).resolve().parent
training_data_file = this_directory / 'askl2_training_data.json'
with open(training_data_file) as fh:
    training_data = json.load(fh)
    fh.seek(0)
    m = hashlib.md5()
    m.update(fh.read().encode('utf8'))
training_data_hash = m.hexdigest()[:10]
sklearn_version = sklearn.__version__
autosklearn_version = autosklearn.__version__
selector_file = pathlib.Path(
    os.environ.get(
        'XDG_CACHE_HOME',
        '~/.cache/auto-sklearn/askl2_selector_%s_%s_%s.pkl'
        % (autosklearn_version, sklearn_version, training_data_hash),
    )
).expanduser()
metafeatures = pd.DataFrame(training_data['metafeatures'])
y_values = np.array(training_data['y_values'])
strategies = training_data['strategies']
minima_for_methods = training_data['minima_for_methods']
maxima_for_methods = training_data['maxima_for_methods']
if not selector_file.exists():
    selector = autosklearn.experimental.selector.OneVSOneSelector(
        configuration=training_data['configuration'],
        default_strategy_idx=strategies.index('RF_SH-eta4-i_holdout_iterative_es_if'),
        rng=1,
    )
    selector.fit(
        X=metafeatures,
        y=y_values,
        methods=strategies,
        minima=minima_for_methods,
        maxima=maxima_for_methods,
    )
    selector_file.parent.mkdir(exist_ok=True, parents=True)
    with open(selector_file, 'wb') as fh:
        pickle.dump(selector, fh)


class SmacObjectCallback:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def __call__(
        self,
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        metalearning_configurations,
        n_jobs,
        dask_client,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario
        from smac.intensification.simple_intensifier import SimpleIntensifier

        scenario = Scenario(scenario_dict)

        initial_configurations = [
            Configuration(configuration_space=scenario.cs, values=member)
            for member in self.portfolio.values()]

        rh2EPM = RunHistory2EPM4LogCost
        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            intensifier=SimpleIntensifier,
            run_id=seed,
            n_jobs=n_jobs,
            dask_client=dask_client,
        )


class SHObjectCallback:
    def __init__(self, budget_type, eta, initial_budget, portfolio):
        self.budget_type = budget_type
        self.eta = eta
        self.initial_budget = initial_budget
        self.portfolio = portfolio

    def __call__(
        self,
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        metalearning_configurations,
        n_jobs,
        dask_client,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario = Scenario(scenario_dict)
        initial_configurations = [
            Configuration(configuration_space=scenario.cs, values=member)
            for member in self.portfolio.values()]

        rh2EPM = RunHistory2EPM4LogCost
        ta_kwargs['budget_type'] = self.budget_type

        smac4ac = SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=SuccessiveHalving,
            intensifier_kwargs={
                'initial_budget': self.initial_budget,
                'max_budget': 100,
                'eta': self.eta,
                'min_chall': 1,
            },
            dask_client=dask_client,
            n_jobs=n_jobs,
        )
        smac4ac.solver.epm_chooser.min_samples_model = int(
            len(scenario.cs.get_hyperparameters()) / 2
        )
        return smac4ac


class AutoSklearn2Classifier(AutoSklearnClassifier):

    def __init__(
        self,
        time_left_for_this_task: int = 3600,
        ensemble_size: int = 50,
        ensemble_nbest: Union[float, int] = 50,
        max_models_on_disc: int = 50,
        seed: int = 1,
        memory_limit: int = 3072,
        tmp_folder: Optional[str] = None,
        output_folder: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        n_jobs: Optional[int] = None,
        dask_client: Optional[dask.distributed.Client] = None,
        disable_evaluator_output: bool = False,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        metric: Optional[Scorer] = None,
        scoring_functions: Optional[List[Scorer]] = None,
        load_models: bool = True,
    ):

        """
        Parameters
        ----------
        time_left_for_this_task : int, optional (default=3600)
            Time limit in seconds for the search of appropriate
            models. By increasing this value, *auto-sklearn* has a higher
            chance of finding better models.

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

        tmp_folder : string, optional (None)
            folder to store configuration output and log files, if ``None``
            automatically use ``/tmp/autosklearn_tmp_$pid_$random_number``

        output_folder : string, optional (None)
            folder to store predictions for optional test set, if ``None``
            no output will be generated

        delete_tmp_folder_after_terminate: string, optional (True)
            remove tmp_folder, when finished. If tmp_folder is None
            tmp_dir will always be deleted

        delete_output_folder_after_terminate: bool, optional (True)
            remove output_folder, when finished. If output_folder is None
            output_dir will always be deleted

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
            `SMAC documentation 
            <https://automl.github.io/SMAC3/master/options.html?highlight=scenario
            #scenario>`_
            for a list of available arguments.

        logging_config : dict, optional (None)
            dictionary object specifying the logger configuration. If None,
            the default logging.yaml file is used, which can be found in
            the directory ``util/logging.yaml`` relative to the installation.

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

        Attributes
        ----------

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            Not all keys returned by scikit-learn are supported yet.

        """  # noqa (links are too long)

        include_estimators = [
            'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp',
        ]
        include_preprocessors = ["no_preprocessing"]
        super().__init__(
            time_left_for_this_task=time_left_for_this_task,
            initial_configurations_via_metalearning=0,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            seed=seed,
            memory_limit=memory_limit,
            include_estimators=include_estimators,
            exclude_estimators=None,
            include_preprocessors=include_preprocessors,
            exclude_preprocessors=None,
            resampling_strategy=None,
            resampling_strategy_arguments=None,
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            n_jobs=n_jobs,
            dask_client=dask_client,
            disable_evaluator_output=disable_evaluator_output,
            get_smac_object_callback=None,
            smac_scenario_args=smac_scenario_args,
            logging_config=logging_config,
            metadata_directory=None,
            metric=metric,
            scoring_functions=scoring_functions,
            load_models=load_models,
        )

    def fit(self, X, y,
            X_test=None,
            y_test=None,
            metric=None,
            feat_type=None,
            dataset_name=None):

        with open(selector_file, 'rb') as fh:
            selector = pickle.load(fh)

        metafeatures = np.array([len(np.unique(y)), X.shape[1], X.shape[0]])
        selection = np.argmax(selector.predict(metafeatures))
        automl_policy = strategies[selection]

        setting = {
            'RF_None_holdout_iterative_es_if': {
                'resampling_strategy': 'holdout-iterative-fit',
                'fidelity': None,
            },
            'RF_None_3CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 3,
                'fidelity': None,
            },
            'RF_None_5CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 5,
                'fidelity': None,
            },
            'RF_None_10CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 10,
                'fidelity': None,
            },
            'RF_SH-eta4-i_holdout_iterative_es_if': {
                'resampling_strategy': 'holdout-iterative-fit',
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_3CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 3,
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_5CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 5,
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_10CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 10,
                'fidelity': 'SH',
            }
        }[automl_policy]

        resampling_strategy = setting['resampling_strategy']
        if resampling_strategy == 'cv-iterative-fit':
            resampling_strategy_kwargs = {'folds': setting['folds']}
        else:
            resampling_strategy_kwargs = None

        portfolio_file = this_directory / 'askl2_portfolios' / ('%s.json' % automl_policy)
        with open(portfolio_file) as fh:
            portfolio_json = json.load(fh)
        portfolio = portfolio_json['portfolio']

        if setting['fidelity'] == 'SH':
            smac_callback = SHObjectCallback('iterations', 4, 5.0, portfolio)
        else:
            smac_callback = SmacObjectCallback(portfolio)

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_arguments = resampling_strategy_kwargs
        self.get_smac_object_callback = smac_callback
        return super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )
