from argparse import ArgumentParser
import os

import arff

from autosklearn.constants import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("metafeatures_dir", help="Output directory of the "
                                                 "metafeature calculation "
                                                 "script.")
    parser.add_argument("results_dir", help="Output directory of the collect "
                                            "results script.")
    parser.add_argument("output_directory", help="Output directory where the "
                                                 "ASLIB scenario will be "
                                                 "written.")
    parser.add_argument("scenario_id", type=str)
    parser.add_argument("algorithm_cutoff_time", type=int)
    parser.add_argument("algorithm_cutoff_memory", type=int)

    args = parser.parse_args()
    metafeatures_dir = args.metafeatures_dir
    results_dir = args.results_dir
    output_directory = args.output_directory

    scenario_id = args.scenario_id
    algorithm_cutoff_time = args.algorithm_cutoff_time
    algorithm_cutoff_memory = args.algorithm_cutoff_memory

    # Create the output directory if necessary
    try:
        os.makedirs(args.output_directory)
    except (OSError, IOError):
        pass

    for sparse, task in [#(1, BINARY_CLASSIFICATION),
                         #(1, MULTICLASS_CLASSIFICATION),
                         #(0, BINARY_CLASSIFICATION),
                         #(0, MULTICLASS_CLASSIFICATION),]:
                         (1, REGRESSION),
                         (0, REGRESSION)]:

        for metric in ['acc_metric', 'auc_metric', 'bac_metric', 'f1_metric',
                       'pac_metric', 'a_metric', 'r2_metric']:

            if STRING_TO_METRIC[metric] not in REGRESSION_METRICS and task in \
                    REGRESSION_TASKS:
                continue
            if STRING_TO_METRIC[metric] not in CLASSIFICATION_METRICS and \
                            task in CLASSIFICATION_TASKS:
                continue

            dir_name = '%s_%s_%s' % (metric, TASK_TYPES_TO_STRING[task],
                'sparse' if sparse else 'dense')
            output_dir_ = os.path.join(output_directory, dir_name)
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
            with open(os.path.join(metafeatures_dir, "description.features.txt")) as fh:
                description_metafeatures = fh.read()

            with open(os.path.join(results_dir_, "description.results.txt")) as fh:
                description_results = fh.read()

            description = [description_metafeatures, description_results]
            description.append("scenario_id: %s" % scenario_id)
            description.append("maximize: false")
            description.append("algorithm_cutoff_time: %d" % algorithm_cutoff_time)
            description.append("algorithm_cutoff_memory: %d" % algorithm_cutoff_memory)

            with open(os.path.join(output_dir_, "description.txt"), "w") as fh:
                for line in description:
                    fh.write(line)
                    fh.write("\n")

            # Copy feature values and add instance id
            with open(os.path.join(metafeatures_dir, "feature_values.arff")) as fh:
                feature_values = arff.load(fh)

            feature_values['relation'] = scenario_id + "_" + feature_values['relation']

            with open(os.path.join(output_dir_, "feature_values.arff"), "w") as fh:
                arff.dump(feature_values, fh)

            # Copy feature runstatus and add instance id
            with open(os.path.join(metafeatures_dir, "feature_runstatus.arff")) as fh:
                feature_runstatus = arff.load(fh)

            feature_runstatus['relation'] = scenario_id + "_" + feature_runstatus['relation']

            with open(os.path.join(output_dir_, "feature_runstatus.arff"), "w") \
                    as fh:
                arff.dump(feature_runstatus, fh)

            # Copy feature runstatus and add instance id
            with open(os.path.join(metafeatures_dir, "feature_costs.arff")) as fh:
                feature_costs = arff.load(fh)

            feature_costs['relation'] = scenario_id + "_" + feature_costs['relation']
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

            algorithm_runs['relation'] = scenario_id + "_" + algorithm_runs['relation']

            with open(os.path.join(output_dir_, "algorithm_runs.arff"), "w") \
                    as fh:
                arff.dump(algorithm_runs, fh)

            # Copy configurations file
            with open(os.path.join(results_dir_, "configurations.csv")) as fh:
                algorithm_runs = fh.read()
            with open(os.path.join(output_dir_, "configurations.csv"), "w") \
                    as fh:
                fh.write(algorithm_runs)