import argparse
import os
import sys

import pyMetaLearn.directory_manager
from pyMetaLearn.entities.task import Task
import pyMetaLearn.entities.entities_base as entities_base


def calculate_metafeatures(directory, tasks):
    pyMetaLearn.directory_manager.set_local_directory(directory)
    available_tasks = entities_base.get_task_list()

    for task in tasks:
        dataset = task._get_dataset()

        print "Calculate the metafeatures for dataset %d, %s" % (
            dataset._id, dataset._name)
        print "Whole dataset...",
        dataset.get_metafeatures(task.target_feature)
        print "and the training set."
        dataset.get_metafeatures(task.target_feature,
            split_file_name=task.estimation_procedure['local_test_split_file'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=sys.maxint)
    parser.add_argument("--ids", type=int, nargs="*", default=[])
    parser.add_argument("directory", type=str)
    args = parser.parse_args()

    pyMetaLearn.directory_manager.set_local_directory(args.directory)
    available_tasks = entities_base.get_task_list()

    tasks = []
    for task_file in available_tasks:
        task = Task.from_file(task_file)
        if (task.task_id > args.min and task.task_id < args.max):
            if args.ids:
                if task.task_id in args.ids:
                    tasks.append(task)
                # Don't append if the task id is not specified
            else:
                tasks.append(task)

    calculate_metafeatures(args.directory, tasks)

