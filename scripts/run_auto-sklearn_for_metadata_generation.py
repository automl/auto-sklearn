import argparse
import json
import logging
import os
import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.constants import BAC_METRIC, R2_METRIC, STRING_TO_METRIC
from autosklearn.evaluation import ExecuteTaFuncWithQueue

from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType

sys.path.append('.')
from update_metadata_util import load_task


parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--time-limit', type=int, required=True)
parser.add_argument('--per-run-time-limit', type=int, required=True)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('--task-type', choices=['classification', 'regression'],
                    required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--unittest', action='store_true')
args = parser.parse_args()

working_directory = args.working_directory
time_limit = args.time_limit
per_run_time_limit = args.per_run_time_limit
task_id = args.task_id
task_type = args.task_type
seed = args.seed
is_test = args.unittest

configuration_output_dir = os.path.join(working_directory, 'configuration',
                                        task_type)
try:
    os.makedirs(configuration_output_dir)
except:
    pass
tmp_dir = os.path.join(configuration_output_dir, '%d-%d' % (task_id, seed))

automl_arguments = {'time_left_for_this_task': time_limit,
                    'per_run_time_limit': per_run_time_limit,
                    'initial_configurations_via_metalearning': 0,
                    'ensemble_size': 0,
                    'ensemble_nbest': 0,
                    'seed': seed,
                    'ml_memory_limit': 3072,
                    'resampling_strategy': 'partial-cv',
                    'resampling_strategy_arguments': {'folds': 10},
                    'delete_tmp_folder_after_terminate': False,
                    'tmp_folder': tmp_dir}

X_train, y_train, X_test, y_test, cat = load_task(task_id)

if task_type == 'classification':
    automl = AutoSklearnClassifier(**automl_arguments)
    metric = BAC_METRIC
elif task_type == 'regression':
    automl = AutoSklearnRegressor(**automl_arguments)
    metric = R2_METRIC
else:
    raise ValueError(task_type)

automl.fit(X_train, y_train, dataset_name=str(task_id), metric=BAC_METRIC,
           feat_type=cat)
data = automl._automl._automl._backend.load_datamanager()
# Data manager can't be replaced with save_datamanager, it has to be deleted
# first
os.remove(automl._automl._automl._backend._get_datamanager_pickle_filename())
data.data['X_test'] = X_test
data.data['Y_test'] = y_test
automl._automl._automl._backend.save_datamanager(data)
trajectory = automl.trajectory_

incumbent_id_to_model = {}
incumbent_id_to_performance = {}
validated_trajectory = []

if is_test:
    memory_limit_factor = 1
else:
    memory_limit_factor = 2

for entry in trajectory:
    incumbent_id = entry[1]
    train_performance = entry[0]
    if incumbent_id not in incumbent_id_to_model:
        config = entry[2]

        logger = logging.getLogger('Testing:)')
        stats = Stats(Scenario({'cutoff_time': per_run_time_limit * 2}))
        stats.start_timing()
        # To avoid the output "first run crashed"...
        stats.ta_runs += 1
        ta = ExecuteTaFuncWithQueue(backend=automl._automl._automl._backend,
                                    autosklearn_seed=seed,
                                    resampling_strategy='test',
                                    with_predictions=False,
                                    memory_limit=memory_limit_factor * automl_arguments['ml_memory_limit'],
                                    disable_file_output=True,
                                    logger=logger,
                                    stats=stats,
                                    all_scoring_functions=True)
        status, cost, runtime, additional_run_info = ta.start(config=config,
                                                              instance=None,
                                                              cutoff=per_run_time_limit)
        if status == StatusType.SUCCESS:
            scores = additional_run_info.split(';')
            scores = [score.split(':') for score in scores]
            scores = [(score[0].strip(), score[1].strip()) for score in scores]
            scores = [(STRING_TO_METRIC[score[0]], score[1]) for score in scores
                      if score[0] in STRING_TO_METRIC]
            scores = {score[0]: float(score[1]) for score in scores}
            assert len(scores) > 1, scores

        # print(additional_run_info)

        validated_trajectory.append(list(entry) + [scores])

validated_trajectory = [entry[:2] + [entry[2].get_dictionary()] + entry[3:]
                        for entry in validated_trajectory]
validated_trajectory_file = os.path.join(tmp_dir, 'validation_trajectory.json')
with open(validated_trajectory_file, 'w') as fh:
    json.dump(validated_trajectory, fh)
