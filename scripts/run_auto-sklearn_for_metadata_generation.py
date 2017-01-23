import argparse
import json
import os
import sys

import scipy.sparse

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.constants import BAC_METRIC, MULTILABEL_CLASSIFICATION, \
    MULTICLASS_CLASSIFICATION, CLASSIFICATION_TASKS, REGRESSION_TASKS, \
    R2_METRIC
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline
from autosklearn.evaluation.util import calculate_score

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
args = parser.parse_args()

working_directory = args.working_directory
time_limit = args.time_limit
per_run_time_limit = args.per_run_time_limit
task_id = args.task_id
task_type = args.task_type
seed = args.seed

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
trajectory = automl.trajectory_

incumbent_id_to_model = {}
incumbent_id_to_performance = {}
validated_trajectory = []

for entry in trajectory:
    incumbent_id = entry[1]
    if incumbent_id not in incumbent_id_to_model:
        config = entry[2]
        print(incumbent_id, config)
        is_sparse = 1 if scipy.sparse.issparse(X_train) else 0
        dataset_properties = {'task': automl._automl._automl._task,
                              'sparse': is_sparse,
                              'is_multilabel': automl._automl._automl._task == MULTILABEL_CLASSIFICATION,
                              'is_multiclass': automl._automl._automl._task == MULTICLASS_CLASSIFICATION}
        init_params = {'one_hot_encoding:categorical_features':
                        [True if c == 'categorical' else False for c in cat]}

        model_arguments = {'dataset_properties': dataset_properties,
                           'init_params': init_params,
                           'random_state': 1,
                           'config': config}

        if automl._automl._automl._task in CLASSIFICATION_TASKS:
            model = SimpleClassificationPipeline(**model_arguments)
        elif automl._automl._automl._task in REGRESSION_TASKS:
            model = SimpleRegressionPipeline(**model_arguments)
        else:
            raise ValueError(automl._automl._automl._task)

        model.fit(X_train, y_train)
        incumbent_id_to_model[incumbent_id] = model
        y_pred = model.predict_proba(X_test)
        scores = calculate_score(metric=metric,
                                 solution=y_test,
                                 prediction=y_pred,
                                 task_type=automl._automl._automl._task,
                                 all_scoring_functions=True,
                                 num_classes=None)
        print(scores)

        validated_trajectory.append(list(entry) + [scores])

validated_trajectory = [entry[:2] + [entry[2].get_dictionary()] + entry[3:]
                        for entry in validated_trajectory]
validated_trajectory_file = os.path.join(tmp_dir, 'validation_trajectory.json')
with open(validated_trajectory_file, 'w') as fh:
    json.dump(validated_trajectory, fh)
