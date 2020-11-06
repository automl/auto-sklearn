import hashlib
import json
import os
import pathlib
import pickle
from typing import Any, Dict, Optional, Union

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
    ):

        include_estimators = [
            'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting',
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
