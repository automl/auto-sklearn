from argparse import ArgumentParser
import itertools
import os

import arff

from autosklearn.constants import *
from autosklearn.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--working-directory", type=str, required=True)
    parser.add_argument("--scenario_id", type=str, default='auto-sklearn')
    parser.add_argument("--algorithm_cutoff_time", type=int, default=1800)
    parser.add_argument("--algorithm_cutoff_memory", type=int, default=3072)

    args = parser.parse_args()
    working_directory = args.working_directory

    output_dir = os.path.join(working_directory, 'metadata')
    results_dir = os.path.join(working_directory, 'configuration_results')
    metafeatures_dir = os.path.join(working_directory, 'metafeatures')

    scenario_id = args.scenario_id
    algorithm_cutoff_time = args.algorithm_cutoff_time
    algorithm_cutoff_memory = args.algorithm_cutoff_memory

    # Create the output directory if necessary
    try:
        os.makedirs(output_dir)
    except (OSError, IOError):
        pass

    for task_type in ('classification', 'regression'):
        if task_type == 'classification':
            metadata_sets = itertools.product(
                [0, 1], [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION],
                CLASSIFICATION_METRICS)
        elif task_type == 'regression':
            metadata_sets = itertools.product(
                [0, 1], [REGRESSION], REGRESSION_METRICS)
        else:
            raise ValueError(task_type)

        input_directory = os.path.join(working_directory, 'configuration', task_type)
        metafeatures_dir_for_task = os.path.join(metafeatures_dir, task_type)

        for sparse, task, metric in metadata_sets:
            print(TASK_TYPES_TO_STRING[task], metric, sparse)

            dir_name = '%s_%s_%s' % (metric, TASK_TYPES_TO_STRING[task],
                                     'sparse' if sparse else 'dense')
            output_dir_ = os.path.join(output_dir, dir_name)
            results_dir_ = os.path.join(results_dir, dir_name)

            if not os.path.exists(results_dir_):
                print("Results directory %s does not exist!" % results_dir_)
                continue

            try:
                os.makedirs(output_dir_)
            except Exception:
                pass

            # Create a readme.txt
            with open(os.path.join(output_dir_, "readme.txt"), "w") as fh:
                pass

            # Create description.txt
            with open(os.path.join(metafeatures_dir_for_task,
                                   "description.features.txt")) as fh:
                description_metafeatures = fh.read()

            with open(os.path.join(results_dir_,
                                   "description.results.txt")) as fh:
                description_results = fh.read()

            description = [description_metafeatures, description_results]
            description.append("scenario_id: %s" % scenario_id)
            description.append("maximize: false")
            description.append(
                "algorithm_cutoff_time: %d" % algorithm_cutoff_time)
            description.append(
                "algorithm_cutoff_memory: %d" % algorithm_cutoff_memory)

            with open(os.path.join(output_dir_, "description.txt"), "w") as fh:
                for line in description:
                    fh.write(line)
                    fh.write("\n")

            # Copy feature values and add instance id
            with open(os.path.join(metafeatures_dir_for_task,
                                   "feature_values.arff")) as fh:
                feature_values = arff.load(fh)

            feature_values['relation'] = scenario_id + "_" + feature_values[
                'relation']

            with open(os.path.join(output_dir_, "feature_values.arff"),
                      "w") as fh:
                arff.dump(feature_values, fh)

            # Copy feature runstatus and add instance id
            with open(os.path.join(metafeatures_dir_for_task,
                                   "feature_runstatus.arff")) as fh:
                feature_runstatus = arff.load(fh)

            feature_runstatus['relation'] = scenario_id + "_" + \
                                            feature_runstatus['relation']

            with open(os.path.join(output_dir_, "feature_runstatus.arff"), "w") \
                    as fh:
                arff.dump(feature_runstatus, fh)

            # Copy feature runstatus and add instance id
            with open(
                    os.path.join(metafeatures_dir_for_task, "feature_costs.arff")) as fh:
                feature_costs = arff.load(fh)

            feature_costs['relation'] = scenario_id + "_" + feature_costs[
                'relation']
            for i in range(len(feature_costs['data'])):
                for j in range(2, len(feature_costs['data'][i])):
                    feature_costs['data'][i][j] = \
                        round(feature_costs['data'][i][j], 5)

            with open(os.path.join(output_dir_, "feature_costs.arff"), "w") \
                    as fh:
                arff.dump(feature_costs, fh)

            # Copy algorithm runs and add instance id
            with open(os.path.join(results_dir_, "algorithm_runs.arff")) as fh:
                algorithm_runs = arff.load(fh)

            algorithm_runs['relation'] = scenario_id + "_" + algorithm_runs[
                'relation']

            with open(os.path.join(output_dir_, "algorithm_runs.arff"), "w") \
                    as fh:
                arff.dump(algorithm_runs, fh)

            # Copy configurations file
            with open(os.path.join(results_dir_, "configurations.csv")) as fh:
                algorithm_runs = fh.read()
            with open(os.path.join(output_dir_, "configurations.csv"), "w") \
                    as fh:
                fh.write(algorithm_runs)
