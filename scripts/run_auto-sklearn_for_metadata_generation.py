import argparse
import json
import logging
import os
import shutil
import sys
import tempfile

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.metrics import accuracy, balanced_accuracy, roc_auc, log_loss, r2, \
    mean_squared_error, mean_absolute_error, root_mean_squared_error

from smac.runhistory.runhistory import RunInfo
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import StatusType

sys.path.append('.')
from update_metadata_util import load_task


parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--time-limit', type=int, required=True)
parser.add_argument('--per-run-time-limit', type=int, required=True)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('--metric', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--unittest', action='store_true')
args = parser.parse_args()

working_directory = args.working_directory
time_limit = args.time_limit
per_run_time_limit = args.per_run_time_limit
task_id = args.task_id
seed = args.seed
metric = args.metric
is_test = args.unittest

X_train, y_train, X_test, y_test, cat, task_type = load_task(task_id)

configuration_output_dir = os.path.join(working_directory, 'configuration',
                                        task_type)
try:
    os.makedirs(configuration_output_dir)
except:
    pass
tmp_dir = os.path.join(configuration_output_dir, str(task_id), metric)

tempdir = tempfile.mkdtemp()
autosklearn_directory = os.path.join(tempdir, "dir")

automl_arguments = {
    'time_left_for_this_task': time_limit,
    'per_run_time_limit': per_run_time_limit,
    'initial_configurations_via_metalearning': 0,
    'ensemble_size': 0,
    'ensemble_nbest': 0,
    'seed': seed,
    'ml_memory_limit': 3072,
    'resampling_strategy': 'partial-cv',
    'delete_tmp_folder_after_terminate': False,
    'tmp_folder': autosklearn_directory,
    'disable_evaluator_output': True,
}

if is_test:
    automl_arguments['resampling_strategy_arguments'] = {'folds': 2}
    if task_type == 'classification':
        automl_arguments['include_estimators'] = ['sgd']
        include = {'classifier': ['sgd']}
    elif task_type == 'regression':
        automl_arguments['include_estimators'] = ['extra_trees']
        automl_arguments['include_preprocessors'] = ['no_preprocessing']
        include = {'regressor': ['extra_trees'], 'feature_preprocessor': ['no_preprocessing']}
    else:
        raise ValueError('Unsupported task type: %s' % str(task_type))
else:
    automl_arguments['resampling_strategy_arguments'] = {'folds': 10}
    include = None

metric = {
    'accuracy': accuracy,
    'balanced_accuracy': balanced_accuracy,
    'roc_auc': roc_auc,
    'logloss': log_loss,
    'r2': r2,
    'mean_squared_error': mean_squared_error,
    'root_mean_squared_error': root_mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
}[metric]
automl_arguments['metric'] = metric

if task_type == 'classification':
    automl = AutoSklearnClassifier(**automl_arguments)
elif task_type == 'regression':
    automl = AutoSklearnRegressor(**automl_arguments)
else:
    raise ValueError(task_type)

automl.fit(X_train, y_train, dataset_name=str(task_id),
           feat_type=cat)
data = automl.automl_._backend.load_datamanager()
# Data manager can't be replaced with save_datamanager, it has to be deleted
# first
os.remove(automl.automl_._backend._get_datamanager_pickle_filename())
data.data['X_test'] = X_test
data.data['Y_test'] = y_test
automl.automl_._backend.save_datamanager(data)
trajectory = automl.trajectory_

incumbent_id_to_model = {}
incumbent_id_to_performance = {}
validated_trajectory = []

if is_test:
    memory_limit_factor = 1
else:
    memory_limit_factor = 2

for entry in trajectory:
    incumbent_id = entry.incumbent_id
    train_performance = entry.train_perf
    if incumbent_id not in incumbent_id_to_model:
        config = entry.incumbent

        logger = logging.getLogger('Testing:)')
        stats = Stats(
            Scenario({
                'cutoff_time': per_run_time_limit * 2,
                'run_obj': 'quality',
            })
        )
        stats.start_timing()
        # To avoid the output "first run crashed"...
        stats.submitted_ta_runs += 1
        stats.finished_ta_runs += 1
        memory_lim = memory_limit_factor * automl_arguments['ml_memory_limit']
        ta = ExecuteTaFuncWithQueue(backend=automl.automl_._backend,
                                    autosklearn_seed=seed,
                                    resampling_strategy='test',
                                    memory_limit=memory_lim,
                                    disable_file_output=True,
                                    logger=logger,
                                    stats=stats,
                                    all_scoring_functions=True,
                                    include=include,
                                    metric=automl_arguments['metric'],
                                    cost_for_crash=get_cost_of_crash(automl_arguments['metric']),
                                    abort_on_first_run_crash=False,)
        run_info, run_value = ta.run_wrapper(
            RunInfo(
                config=config,
                instance=None,
                instance_specific=None,
                seed=1,
                cutoff=per_run_time_limit*3,
                capped=False,
            )
        )

        if run_value.status == StatusType.SUCCESS:
            assert len(run_value.additional_info) > 1, run_value.additional_info

        # print(additional_run_info)

        validated_trajectory.append(list(entry) + [task_id] +
                                    [run_value.additional_info])

validated_trajectory = [entry[:2] + [entry[2].get_dictionary()] + entry[3:]
                        for entry in validated_trajectory]
validated_trajectory_file = os.path.join(autosklearn_directory,
                                         'smac3-output',
                                         'run_%d' % seed,
                                         'validation_trajectory.json')
with open(validated_trajectory_file, 'w') as fh:
    json.dump(validated_trajectory, fh, indent=4)

shutil.copytree(autosklearn_directory, tmp_dir)
try:
    shutil.rmtree(tempdir)
except:
    pass
