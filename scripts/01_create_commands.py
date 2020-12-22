import argparse
import itertools
import os
import sys

import openml

sys.path.append('.')
from update_metadata_util import classification_tasks, regression_tasks


parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
working_directory = args.working_directory
test = args.test

command_file_name = os.path.join(working_directory, 'metadata_commands.txt')

this_directory = os.path.dirname(os.path.abspath(__file__))
script_name = 'run_auto-sklearn_for_metadata_generation.py'
absolute_script_name = os.path.join(this_directory, script_name)

commands = []
for task_id in (classification_tasks if not test else (233, 245, 258)):
    for metric in ('accuracy', 'balanced_accuracy', 'roc_auc', 'logloss'):

        if (
            len(openml.tasks.get_task(task_id, download_data=False).class_labels) > 2
            and metric == 'roc_auc'
        ):
            continue

        command = ('python3 %s --working-directory %s --time-limit 86400 '
                   '--per-run-time-limit 1800 --task-id %d -s 1 --metric %s' %
                   (absolute_script_name, working_directory, task_id, metric))
        commands.append(command)
for task_id in (regression_tasks if not test else (360029, 360033)):
    for metric in ('r2', 'root_mean_squared_error', 'mean_absolute_error'):
        command = ('python3 %s --working-directory %s --time-limit 86400 '
                   '--per-run-time-limit 1800 --task-id %d -s 1 --metric %s' %
                   (absolute_script_name, working_directory, task_id, metric))
        commands.append(command)

with open(command_file_name, 'w') as fh:
    for command in commands:
        fh.writelines(command)
        fh.write('\n')
