from argparse import ArgumentParser
from collections import defaultdict
import csv
import glob
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
    outputs = dict()
    configurations = dict()
    configurations_to_ids = dict()

    try:
        validation_trajectory_files = glob.glob(os.path.join(
            validation_directory, '*', '*', 'validation_trajectory_*.json'
        ))
    except FileNotFoundError:
        return {}, {}

    for validation_trajectory_file in validation_trajectory_files:
        task_name = None

        with open(validation_trajectory_file) as fh:
            validation_trajectory = json.load(fh)

        best_value = np.inf
        best_configuration = None
        best_configuration_dir = None

        n_configs = 0
        n_better = 0
        n_broken = 0

        for entry in validation_trajectory:
            # There's no reason to keep the default configuration
            # (even if it's better) because it is run anyway
            if validation_trajectory[0][2] == entry[2]:
                continue

            n_configs += 1

            config = entry[2]
            task_name = entry[-2]
            score = entry[-1].get(str(metric), np.inf)

            if np.isinf(score) and np.isinf(best_value) or score < best_value:
                n_better += 1

                try:
                    best_configuration = Configuration(
                        configuration_space=configuration_space, values=config)
                    best_value = score
                    best_configuration_dir = validation_trajectory_file
                except Exception as e:
                    print(e)
                    n_broken += 1

        if task_name is None:
            print('Could not find any configuration better than the default configuration!')
            continue

        if best_configuration is None:
            print('Could not find a valid configuration; total %d, better %d, broken %d'
                  % (n_configs, n_better, n_broken))
            continue
        elif best_configuration in configurations_to_ids:
            print('Found configuration in', best_configuration_dir)
            config_id = configurations_to_ids[best_configuration]
        else:
            print('Found configuration in', best_configuration_dir)
            config_id = len(configurations_to_ids)
            configurations_to_ids[config_id] = best_configuration
            configurations[config_id] = best_configuration

        # We could keep multiple configurations per task (and actually did so before), but
        # there is really no reason to already filter them here and only keep the best
        # (this is less confusing when looking at the raw data later on).
        if task_name not in outputs:
            outputs[task_name] = (config_id, best_value)
        else:
            if best_value < outputs[task_name][1]:
                outputs[task_name] = (config_id, best_value)

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
    keep_configurations = set()
    for dataset, (configuration_id, value) in outputs.items():

        if not np.isfinite(value):
            runstatus = 'not_applicable'
            value = None
        else:
            runstatus = 'ok'

        line = [dataset, 1, configuration_id + 1, value, runstatus]
        data.append(line)
        keep_configurations.add(configuration_id)

    arff_object['data'] = data

    with open(os.path.join(output_dir, "algorithm_runs.arff"), "w") as fh:
        arff.dump(arff_object, fh)

    hyperparameters = []
    for idx in configurations:
        if idx not in keep_configurations:
            continue
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
    parser.add_argument("--cutoff", type=int, default=-1)
    parser.add_argument("--only-best", type=bool, default=True)

    args = parser.parse_args()
    working_directory = args.working_directory
    cutoff = args.cutoff
    only_best = args.only_best

    for task_type in ('classification', 'regression'):
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
