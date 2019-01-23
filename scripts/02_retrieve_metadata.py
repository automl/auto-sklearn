from argparse import ArgumentParser
from collections import defaultdict
import csv
import itertools
import json
import os

import arff
import numpy as np

from ConfigSpace.configuration_space import Configuration

from autosklearn.constants import *
from autosklearn.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from autosklearn.util import pipeline


def retrieve_matadata(validation_directory, metric, configuration_space,
                      cutoff=0, only_best=True):
    if not only_best:
        raise NotImplementedError()
    if cutoff > 0:
        raise NotImplementedError()

    # Mapping from task id to a list of (config, score) tuples
    outputs = defaultdict(list)
    configurations = dict()
    configurations_to_ids = dict()

    possible_experiment_directories = os.listdir(validation_directory)

    for ped in possible_experiment_directories:
        task_name = None

        ped = os.path.join(validation_directory, ped)
        if not os.path.exists(ped) or not os.path.isdir(ped):
            continue
        print("Going through directory %s" % ped)

        validation_trajectory_file = os.path.join(ped, 'smac3-output', 'run_1',
                                                  'validation_trajectory.json')
        with open(validation_trajectory_file) as fh:
            validation_trajectory = json.load(fh)

        best_value = np.inf
        best_configuration = None
        for entry in validation_trajectory:
            config = entry[2]
            task_name = entry[-2]
            score = entry[-1].get(str(metric), np.inf)

            if score < best_value:
                try:
                    best_configuration = Configuration(
                        configuration_space=configuration_space, values=config)
                    best_value = score
                except:
                    pass

        if task_name is None:
            continue

        if best_configuration is None:
            continue
        elif best_configuration in configurations_to_ids:
            config_id = configurations_to_ids[best_configuration]
        else:
            config_id = len(configurations_to_ids)
            configurations_to_ids[config_id] = best_configuration
            configurations[config_id] = best_configuration

        outputs[task_name].append((config_id, best_value))

    return outputs, configurations


def write_output(outputs, configurations, output_dir, configuration_space,
                 metric):
    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC'),
                                 ('algorithm', 'STRING'),
                                 (metric, 'NUMERIC'),
                                 ('runstatus',
                                  ['ok', 'timeout', 'memout', 'not_applicable',
                                   'crash', 'other'])]
    arff_object['relation'] = "ALGORITHM_RUNS"
    arff_object['description'] = ""

    data = []
    for dataset in outputs:
        for configuration_id, value in outputs[dataset]:

            if not np.isfinite(value):
                runstatus = 'not_applicable'
                value = None
            else:
                runstatus = 'ok'

            line = [dataset, 1, configuration_id + 1, value, runstatus]
            data.append(line)
    arff_object['data'] = data

    with open(os.path.join(output_dir, "algorithm_runs.arff"), "w") as fh:
        arff.dump(arff_object, fh)

    hyperparameters = []
    for idx in configurations:
        configuration = configurations[idx]
        line = {'idx': idx + 1}
        for hp_name in configuration:
            value = configuration[hp_name]
            if value is not None:
                line[hp_name] = value

        hyperparameters.append(line)

    fieldnames = ['idx']
    for hyperparameter in configuration_space.get_hyperparameters():
        fieldnames.append(hyperparameter.name)
    fieldnames = [fieldnames[0]] + sorted(fieldnames[1:])
    with open(os.path.join(output_dir, "configurations.csv"), "w") as fh:
        csv_writer = csv.DictWriter(fh, fieldnames=fieldnames)
        csv_writer.writeheader()
        for line in hyperparameters:
            csv_writer.writerow(line)

    description = dict()
    description['algorithms_deterministic'] = \
        ",".join([str(configuration_id + 1)
                  for configuration_id in sorted(configurations.keys())])
    description['algorithms_stochastic'] = \
        ",".join([])
    description['performance_measures'] = metric
    description['performance_type'] = 'solution_quality'

    with open(os.path.join(output_dir, "description.results.txt"),
              "w") as fh:
        for key in description:
            fh.write("%s: %s\n" % (key, description[key]))


def main():
    parser = ArgumentParser()

    parser.add_argument("--working-directory", type=str, required=True)
    parser.add_argument("--task-type", required=True,
                        choices=['classification', 'regression'])
    parser.add_argument("--cutoff", type=int, default=-1)
    parser.add_argument("--only-best", type=bool, default=True)

    args = parser.parse_args()
    working_directory = args.working_directory
    task_type = args.task_type
    cutoff = args.cutoff
    only_best = args.only_best

    if task_type == 'classification':
        metadata_sets = itertools.product(
            [0, 1], [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION],
            CLASSIFICATION_METRICS)
        input_directory = os.path.join(working_directory, 'configuration',
                                       'classification')
    elif task_type == 'regression':
        metadata_sets = itertools.product(
            [0, 1], [REGRESSION], REGRESSION_METRICS)
        input_directory = os.path.join(working_directory, 'configuration',
                                       'regression')
    else:
        raise ValueError(task_type)

    output_dir = os.path.join(working_directory, 'configuration_results')

    for sparse, task, metric in metadata_sets:
        print(TASK_TYPES_TO_STRING[task], metric, sparse)

        output_dir_ = os.path.join(output_dir, '%s_%s_%s' % (
            metric, TASK_TYPES_TO_STRING[task],
            'sparse' if sparse else 'dense'))

        configuration_space = pipeline.get_configuration_space(
            {'is_sparse': sparse, 'task': task})

        outputs, configurations = retrieve_matadata(
            validation_directory=input_directory,
            metric=metric,
            cutoff=cutoff,
            configuration_space=configuration_space,
            only_best=only_best)

        if len(outputs) == 0:
            print("No output found for %s, %s, %s" %
                  (metric, TASK_TYPES_TO_STRING[task],
                   'sparse' if sparse else 'dense'))
            continue

        try:
            os.makedirs(output_dir_)
        except:
            pass

        write_output(outputs, configurations, output_dir_,
                     configuration_space, metric)


if __name__ == "__main__":
    main()
