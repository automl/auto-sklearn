"""Generate HPOlib experiments to get metadata.

Call:

python automl1_metalearning_experiments.py datasets.csv output path/to/HPOlib/optimizers/smac/smac_2_08

Additional arguments:

--openml path
  Change the OpenML cache directory. Useful for working with local datasets
  which are in the OpenML format
--product bool
  If False, generate an experiment for the whole configuration space applicable
  to a dataset. If True, generate an experiment for each element in the
  product of all classifiers, preprocessors, metrics.
"""


from argparse import ArgumentParser
from collections import defaultdict
import glob
from itertools import product
import os

import numpy as np
from openml.apiconnector import APIConnector

from autosklearn.models.paramsklearn import get_configuration_space
from autosklearn.data.data_converter import predict_RAM_usage
from AutoSklearn.components.classification import _classifiers
from AutoSklearn.components.preprocessing import _preprocessors
from HPOlibConfigSpace.converters import pcs_parser


def filter_datasets(dataset_ids, openml_cache_directory=None):
    rval = []

    connector = APIConnector(cache_directory=openml_cache_directory,
                             authenticate=False)

    for did in dataset_ids:
        dataset = connector.get_cached_dataset(did)

        X, y, categorical = dataset.get_pandas(
            target=dataset.default_target_attribute, include_row_id=False,
            include_ignore_attributes=False)
        X = X.values
        y = y.values
        num_missing_values = np.sum(~np.isfinite(X))
        num_values = X.shape[0] * X.shape[1]
        sparsity = float(num_missing_values) / float(num_values)
        print dataset.id, X.shape, sparsity,

        estimated_ram = float(predict_RAM_usage(X, categorical)) / 1024 / 1024
        print "Num categorical:", np.sum(categorical), estimated_ram
        sparse = False
        if np.sum(categorical) > 0 and estimated_ram > 1000:
            sparse = True

        if np.sum(categorical) == len(categorical):
            print "Do not use dataset %d because it will be empty after removing " \
                  "categorical variables." % dataset.id
            continue

        rval.append({'did': did, 'sparse': sparse})

    return rval


def create_metadata_directories(datasets, metrics, output_dir, smac,
                                function_arguments=None):
    commands = list()

    if function_arguments is None:
        function_arguments = ""

    for dataset_info in datasets:
        did = dataset_info['did']

        for metric in metrics:

            # Create configuration space, we can put everything in here because by
            # default it is not sparse and not multiclass and not multilabel
            try:
                configuration_space = get_configuration_space(
                    {'task': 'binary.classification',
                     'is_sparse': dataset_info['sparse']})
            except (ValueError, KeyError) as e:
                print metric
                print e, e.message
                continue

            # Create the directories after creating the configuration space to
            # avoid creating empty directories!

            experiment_dir = os.path.join(output_dir, "autosklearn")
            try:
                os.mkdir(experiment_dir)
            except (IOError, OSError):
                pass

            output_dir_ = os.path.join(experiment_dir, str(did))

            # If the output dir exists and there is a SMAC run inside,
            # don't add a command
            smac_experiments = glob.glob(os.path.join(output_dir_, "smac_2_*"))
            if len(smac_experiments) > 0:
                continue

            # If the output directory does not exist, create it
            # Otherwise, still add the command to the list of all commands
            if not os.path.exists(output_dir_):
                os.mkdir(output_dir_)

                with open(os.path.join(output_dir_, "params.pcs"), "w") as fh:
                    cs = pcs_parser.write(configuration_space)
                    fh.write(cs)

                with open(os.path.join(output_dir_, "config.cfg"), "w") as fh:
                    write_config_file(fh,
                          HPOLIB={
                              'dispatcher': 'runsolver_wrapper.py',
                              'function':
                                  'python -m ' \
                                  'AutoML2015.wrapper.openml_wrapper ' \
                                  "--dataset %s --metric %s " \
                                  "--task_type %s --remove_categorical %s" \
                                  % (str(did).replace("-", "\-"), metric,
                                     "binary.classification",
                                     function_arguments),
                              'number_of_jobs': 10000,
                              'number_cv_folds': 10,
                              'runsolver_time_limit': 1800,
                              'memory_limit': 4000,
                              'result_on_terminate': 1.0},
                          SMAC={'runtime_limit': 172800, #2 days
                                'p': 'params.pcs'})

            commands.append(
                "HPOlib-run -o %s --cwd %s --HPOLIB:temporary_output_directory "
                "/tmp/${JOB_ID}.${SGE_TASK_ID}.aad_core.q" % (
                    smac, output_dir_))

    for seed in range(1000, 10001, 1000):
        with open(os.path.join(output_dir, "autosklearn_%d.txt" % seed),
                  "w") as fh:
            for command in commands:
                fh.write(command)
                fh.write(" --seed %d --HPOLIB:optimizer_loglevel 10" % seed)
                fh.write("\n")


def create_metadata_directories_product(datasets, metrics, output_dir, smac,
                                        function_arguments=None):
    commands = defaultdict(lambda: defaultdict(list))

    if function_arguments is None:
        function_arguments = ""

    classifiers = _classifiers
    preprocessors = [p for p in _preprocessors if p not in
                     ["imputation", "rescaling"]] + ["None"]

    print "#datasets", len(datasets)
    print "#classifiers", len(classifiers)
    print "#preprocessors", len(preprocessors)
    print "#metrics", len(metrics)
    print "Approximately %d experiment directories will be created." % \
          (len(datasets) * len(classifiers) * len(preprocessors) * len(
           metrics))
    print

    for dataset_info in datasets:
        did = dataset_info['did']

        for classifier, preprocessor, metric in \
                product(classifiers, preprocessors, metrics):

            # Create configuration space, we can put everything in here because by
            # default it is not sparse and not multiclass and not multilabel
            try:
                configuration_space = get_configuration_space(
                    {'task': 'binary.classification',
                     'is_sparse': dataset_info['sparse']},
                    include_estimators=[classifier],
                    include_preprocessors=[preprocessor])
            except (ValueError, KeyError) as e:
                print classifier, preprocessor, metric
                print e, e.message
                continue

            classifiers_ = ["extra_trees", "gradient_boosting",
                            "k_nearest_neighbors", "libsvm_svc",
                            "random_forest"]
            feature_learning_ = ["kitchen_sinks", "sparse_filtering"]
            if classifier in classifiers_ and preprocessor in feature_learning_:
                continue

            if preprocessor != "None":
                pre = configuration_space.get_hyperparameter("preprocessor")
                item_number = pre.choices.index("None")
                del pre.choices[item_number]
                pre.default = preprocessor

                # After removing None, it's possible that the choices are empty
                if len(pre.choices) == 0:
                    continue

            # Create the directories after creating the configuration space to
            # avoid creating empty directories!
            clf_pre_met_combination_dir = "%s-%s-%s" % \
                                          (classifier, preprocessor, metric)
            combination_directory = os.path.join(output_dir,
                                                 clf_pre_met_combination_dir)
            try:
                os.mkdir(combination_directory)
            except (IOError, OSError):
                pass

            output_dir_ = os.path.join(combination_directory, str(did))

            # If the output dir exists and there is a SMAC run inside,
            # don't add a command
            smac_experiments = glob.glob(os.path.join(output_dir_, "smac_2_*"))
            if len(smac_experiments) > 0:
                continue

            # If the output directory does not exist, create it
            # Otherwise, still add the command to the list of all commands
            if not os.path.exists(output_dir_):
                os.mkdir(output_dir_)

                with open(os.path.join(output_dir_, "params.pcs"), "w") as fh:
                    cs = pcs_parser.write(configuration_space)
                    fh.write(cs)

                with open(os.path.join(output_dir_, "config.cfg"), "w") as fh:
                    write_config_file(fh,
                        HPOLIB={'dispatcher': 'runsolver_wrapper.py',
                                'function':
                                    'python -m '
                                    'AutoML2015.wrapper.openml_wrapper '
                                    "--dataset %s --metric %s "
                                    "--task_type %s --remove_categorical %s"
                                    % (str(did).replace("-", "/-"), metric,
                                       "binary.classification",
                                       function_arguments),
                                'number_of_jobs': 10000,
                                'number_cv_folds': 10,
                                'runsolver_time_limit': 1800,
                                'memory_limit': 4000,
                                'result_on_terminate': 1.0},
                        SMAC={'runtime_limit': 172800, 'p': 'params.pcs'})

            commands[classifier][preprocessor].append(
                "HPOlib-run -o %s --cwd %s --HPOLIB:temporary_output_directory "
                "/tmp/${JOB_ID}.${SGE_TASK_ID}.aad_core.q" % (smac, output_dir_))

        for classifier, preprocessor in product(classifiers, preprocessors):
            if len(commands[classifier][preprocessor]) == 0:
                continue
            for seed in range(1000, 10001, 1000):
                with open(os.path.join(output_dir, "%s_%s_%d.txt"
                        % (classifier, preprocessor, seed)), "w") as fh:
                    for command in commands[classifier][preprocessor]:
                        fh.write(command + " --seed %d --HPOLIB:optimizer_loglevel 10" % seed)
                        fh.write("\n")


def write_config_file(fh, **kwargs):
    for section in kwargs:
        fh.write("[%s]\n" % section.upper())

        for key in kwargs[section]:
            fh.write("%s = %s\n" % (key, kwargs[section][key]))
    return fh


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="CSV file describing for which "
                                          "datasets to create an HPOlib "
                                          "experiment directory. Must have "
                                          "one column. Every row then "
                                          "contains one dataset id.")
    parser.add_argument("output_directory", help="Output directory where the "
                                                 "HPOlib experiment "
                                                 "directories will be created.")
    parser.add_argument("path_to_SMAC", help="Path to the HPOlib SMAC plugin.")
    parser.add_argument("--product", help="Create experiments for the product of"
                                          "classifiers, preprocessors and "
                                          "metrics.", type=bool, default=False)
    parser.add_argument("--openml", type=str, default=None)

    args = parser.parse_args()

    dataset_ids = []
    with open(args.input) as fh:
        for row in fh:
            dataset_ids.append(int(float(row.strip())))

    metrics = ["bac_metric"]
    function_arguments = ""
    if args.openml:
        function_arguments += " --openml_cache_directory %s" % \
                              args.openml

    datasets_info = filter_datasets(dataset_ids,
        openml_cache_directory=args.openml)

    if args.product:
        create_metadata_directories_product(datasets_info, metrics,
            args.output_directory, args.path_to_SMAC, function_arguments)
    else:
        create_metadata_directories(datasets_info, metrics,
            args.output_directory, args.path_to_SMAC, function_arguments)
