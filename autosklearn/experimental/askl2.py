import json
import multiprocessing
import os
import pickle
from typing import Any, Dict, Optional, Union

from ConfigSpace import Configuration
import numpy as np
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.experimental.selector
from autosklearn.metrics import Scorer


CALLBACK_COUNTER = multiprocessing.Value('i', 0)


this_directory = os.path.abspath(os.path.dirname(__file__))
selector_file = os.path.join(this_directory, 'selector.pkl')
training_data_file = os.path.join(this_directory, 'askl2_training_data.json')
with open(training_data_file) as fh:
    training_data = json.load(fh)
metafeatures = pd.DataFrame(training_data['metafeatures'])
y_values = np.array(training_data['y_values'])
strategies = training_data['strategies']
minima_for_methods = training_data['minima_for_methods']
maxima_for_methods = training_data['maxima_for_methods']
if not os.path.exists(selector_file):
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
    with open(selector_file, 'wb') as fh:
        pickle.dump(selector, fh)


def get_smac_object_callback(portfolio, lock):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)

        lock.acquire()
        try:
            global CALLBACK_COUNTER
            print(CALLBACK_COUNTER.value, flush=True)
            if CALLBACK_COUNTER.value == 0:
                initial_configurations = [
                    Configuration(configuration_space=scenario.cs, values=member)
                    for member in portfolio.values()]
            else:
                initial_configurations = [scenario.cs.sample_configuration(size=1)]
            CALLBACK_COUNTER.value += 1
        finally:
            lock.release()

        rh2EPM = RunHistory2EPM4LogCost
        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
        )
    return get_smac_object


def get_sh_or_hb_object_callback(budget_type, bandit_strategy, eta, initial_budget, portfolio,
                                 lock):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.intensification.hyperband import Hyperband
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)

        lock.acquire()
        try:
            global CALLBACK_COUNTER
            if CALLBACK_COUNTER.value == 0:
                initial_configurations = [
                    Configuration(configuration_space=scenario.cs, values=member)
                    for member in portfolio.values()]
            else:
                initial_configurations = [scenario.cs.sample_configuration(size=1)]
            CALLBACK_COUNTER.value += 1
        finally:
            lock.release()

        rh2EPM = RunHistory2EPM4LogCost

        ta_kwargs['budget_type'] = budget_type

        if bandit_strategy == 'sh':
            bandit = SuccessiveHalving
        elif bandit_strategy == 'hb':
            bandit = Hyperband
        else:
            raise ValueError(bandit_strategy)

        smac4ac = SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=bandit,
            intensifier_kwargs={
                'initial_budget': initial_budget,
                'max_budget': 100,
                'eta': eta,
                'min_chall': 1},
            )
        smac4ac.solver.epm_chooser.min_samples_model = int(
            len(scenario.cs.get_hyperparameters()) / 2
        )
        return smac4ac
    return get_smac_object


class AutoSklearn2Classifier(AutoSklearnClassifier):

    def __init__(
        self,
        time_left_for_this_task: int = 3600,
        ensemble_size: int = 50,
        ensemble_nbest: Union[float, int] = 50,
        max_models_on_disc: int = 50,
        ensemble_memory_limit: int = 1024,
        seed: int = 1,
        ml_memory_limit: int = 3072,
        tmp_folder: Optional[str] = None,
        output_folder: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        shared_mode: bool = False,
        n_jobs: Optional[int] = None,
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
            ensemble_memory_limit=ensemble_memory_limit,
            seed=seed,
            ml_memory_limit=ml_memory_limit,
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
            shared_mode=shared_mode,
            n_jobs=n_jobs,
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

        portfolio_file = os.path.join(this_directory, 'askl2_portfolios', '%s.json' % automl_policy)
        with open(portfolio_file) as fh:
            portfolio_json = json.load(fh)
        portfolio = portfolio_json['portfolio']

        lock = multiprocessing.Lock()
        if setting['fidelity'] == 'SH':
            smac_callback = get_sh_or_hb_object_callback('iterations', 'sh', 4, 5.0, portfolio,
                                                         lock)
        else:
            smac_callback = get_smac_object_callback(portfolio, lock)

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
