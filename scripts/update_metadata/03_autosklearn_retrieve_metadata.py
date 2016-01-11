from argparse import ArgumentParser
from collections import defaultdict
import csv
import os

import arff
import numpy as np

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import IntegerHyperparameter, \
    FloatHyperparameter, CategoricalHyperparameter, Constant

from autosklearn.constants import *
from autosklearn.util import pipeline


def retrieve_matadata(validation_directory, metric, configuration_space,
                      cutoff=0, num_runs=1, only_best=False):
    # This looks weird! The dictionaries contain the following information
    # {dataset_1: (configuration: best_value), dataset_2: (configuration: # best_value)}
    outputs = defaultdict(list)
    configurations = dict()
    configurations_to_ids = dict()
    possible_experiment_directories = os.listdir(validation_directory)

    for ped in possible_experiment_directories:
        dataset_name = ped

        # This is hacky, replace by pySMAC!
        ped = os.path.join(validation_directory, ped, ped)

        if not os.path.exists(ped) or not os.path.isdir(ped):
            continue

        smac_output_dir = ped
        validation_files = []
        validation_configuration_files = []
        validation_run_results_files = []

        # Configurations from smac-validate from a trajectory file
        for seed in [num_run * 1000 for num_run in range(num_runs)]:
            validation_file = os.path.join(smac_output_dir,
                                           'validationResults-detailed-traj-run-%d-walltime.csv' % seed)
            validation_configuration_file = os.path.join(smac_output_dir,
                                                         'validationCallStrings-detailed-traj-run-%d-walltime.csv' % seed)
            validation_run_results_file = os.path.join(smac_output_dir,
                                                       'validationRunResultLineMatrix-detailed-traj-run-%d-walltime.csv'
                                                       % seed)

            if os.path.exists(validation_file) and os.path.exists(
                    validation_configuration_file) and os.path.exists(
                    validation_run_results_file):
                validation_files.append(validation_file)
                validation_configuration_files.append(
                    validation_configuration_file)
                validation_run_results_files.append(
                    validation_run_results_file)

        # Configurations from smac-validate from a configurations file
        validation_file = os.path.join(smac_output_dir,
                                           'validationResults-configurations-walltime.csv')
        validation_configuration_file = os.path.join(smac_output_dir,
                                                     'validationCallStrings-configurations-walltime.csv')
        validation_run_results_file = os.path.join(smac_output_dir,
                                                   'validationRunResultLineMatrix-configurations-walltime.csv')
        if os.path.exists(validation_file) and os.path.exists(
                validation_configuration_file) and os.path.exists(
                validation_run_results_file):
            validation_files.append(validation_file)
            validation_configuration_files.append(
                validation_configuration_file)
            validation_run_results_files.append(
                validation_run_results_file)

        for validation_file, validation_configuration_file, validation_run_results_file in \
                zip(validation_files, validation_configuration_files,
                    validation_run_results_files):
            configuration_to_time = dict()
            with open(validation_file) as fh:
                reader = csv.reader(fh)
                reader.next()
                for row in reader:
                    current_time = float(row[0])
                    validation_configuration_id = int(row[4])
                    configuration_to_time[
                        validation_configuration_id] = current_time

            best = []
            with open(validation_run_results_file) as fh:
                reader = csv.reader(fh)
                reader.next()
                for row in reader:
                    seed = int(float(row[1]))
                    results = row[2:]
                    for i, result in enumerate(results):
                        result = result.split(",")[-1]
                        if not ";" in result:
                            continue
                        result = result.split(";")
                        for result_ in result:
                            metric_, value = result_.split(":")
                            metric_ = metric_.replace(":", "").strip()
                            value = value.strip()

                            if metric_ == metric:
                                value = float(value)
                                best.append((value, i + 1))

            best.sort()
            for test_performance, validation_configuration_id in best:
                if cutoff > 0 and \
                        configuration_to_time[validation_configuration_id] > \
                        cutoff:
                    continue
                stop = False
                with open(validation_configuration_file) as fh:
                    reader = csv.reader(fh)
                    reader.next()
                    for row in reader:
                        if int(row[0]) == validation_configuration_id:
                            configuration = row[1]
                            configuration = configuration.split()
                            configuration = {configuration[i]:
                                                 configuration[i + 1]
                                             for i in
                                             range(0, len(configuration),
                                                   2)}
                            for key in configuration.keys():
                                value = configuration[key]
                                hp_name = key[1:]
                                try:
                                    hyperparameter = \
                                        configuration_space.get_hyperparameter(
                                            hp_name)
                                except KeyError:
                                    break
                                value = value.strip("'")

                                if isinstance(hyperparameter,
                                              IntegerHyperparameter):
                                    value = int(float(value))
                                elif isinstance(hyperparameter,
                                                FloatHyperparameter):
                                    value = float(value)
                                elif isinstance(hyperparameter,
                                                CategoricalHyperparameter):
                                    # Implementation tailored to the PCS
                                    # parser
                                    value = str(value)
                                elif isinstance(hyperparameter, Constant):
                                    if isinstance(hyperparameter.value, float):
                                        value = float(value)
                                    elif isinstance(hyperparameter.value, int):
                                        value = int(value)
                                    else:
                                        value = value
                                elif hyperparameter is None:
                                    value = ''
                                else:
                                    raise ValueError((hp_name, ))

                                configuration[hp_name] = value

                            try:
                                configuration = Configuration(
                                    configuration_space, configuration)
                            except Exception as e:
                                print("Configuration %s not applicable " \
                                      "because of %s!" \
                                      % (row[1], e))
                                break

                            if str(configuration) in \
                                    configurations_to_ids:
                                global_configuration_id = \
                                    configurations_to_ids[
                                        str(configuration)]
                            else:
                                global_configuration_id = len(configurations)
                                configurations[
                                    global_configuration_id] = configuration
                                configurations_to_ids[str(configuration)] = \
                                    global_configuration_id

                            if global_configuration_id is not None:
                                outputs[dataset_name].append(
                                    (global_configuration_id, test_performance))

                            if only_best:
                                stop = True
                                break
                            else:
                                pass

                if stop is True:
                    break

    return outputs, configurations


def write_output(outputs, configurations, output_dir, configuration_space,
                 metric):
    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC'),
                                 ('algorithm', 'STRING')] + \
                                [(metric, 'NUMERIC')] + \
                                [('runstatus',
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

    parser.add_argument("configuration_directory",
                        metavar="configuration-directory")
    parser.add_argument("output_directory", metavar="output-directory")
    parser.add_argument("--cutoff", type=int, default=-1,
                        help="Only consider the validation performances up to "
                             "this time.")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--only-best", type=bool, default=False,
                        help="Look only for the best configuration in the "
                             "validation files.")

    args = parser.parse_args()
    configuration_directory = args.configuration_directory
    output_dir = args.output_directory
    cutoff = int(args.cutoff)
    num_runs = args.num_runs

    for sparse, task in [(1, BINARY_CLASSIFICATION),
                         (1, MULTICLASS_CLASSIFICATION),
                         (0, BINARY_CLASSIFICATION),
                         (0, MULTICLASS_CLASSIFICATION)]:

        for metric in ['acc_metric', 'auc_metric', 'bac_metric', 'f1_metric',
                       'pac_metric']:

            output_dir_ = os.path.join(output_dir, '%s_%s_%s' % (
                metric, TASK_TYPES_TO_STRING[task], 'sparse' if sparse else 'dense'))

            configuration_space = pipeline.get_configuration_space(
                {'is_sparse': sparse, 'task': task}
            )

            try:
                os.makedirs(output_dir_)
            except:
                pass

            outputs, configurations = retrieve_matadata(
                validation_directory=configuration_directory,
                num_runs=num_runs,
                metric=metric,
                cutoff=cutoff,
                configuration_space=configuration_space,
                only_best=args.only_best)

            if len(outputs) == 0:
                raise ValueError("Nothing found!")

            write_output(outputs, configurations, output_dir_,
                         configuration_space, metric)


if __name__ == "__main__":
    main()

