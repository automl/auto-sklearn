import argparse
import os
import sys

sys.path.append('.')
from update_metadata_util import classification_tasks, regression_tasks


parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--task-type', required=True,
                    choices=['classification', 'regression'])
args = parser.parse_args()
working_directory = args.working_directory
task_type = args.task_type

if task_type == 'classification':
    tasks = classification_tasks
    command_file_name = os.path.join(working_directory, 'umd-cls.txt')
else:
    tasks = regression_tasks
    command_file_name = os.path.join(working_directory, 'umd-rs.txt')

this_directory = os.path.dirname(os.path.abspath(__file__))
script_name = 'run_auto-sklearn_for_metadata_generation.py'
absolute_script_name = os.path.join(this_directory, script_name)

commands = []
for task_id in tasks:
    command = ('python3 %s --working-directory %s --time-limit 86400 '
               '--per-run-time-limit 1800 --task-id %d --task-type %s -s 1' %
               (absolute_script_name, working_directory, task_id, task_type))
    commands.append(command)

with open(command_file_name, 'w') as fh:
    for command in commands:
        fh.writelines(command)
        fh.write('\n')
