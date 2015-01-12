from itertools import product
import os
from StringIO import StringIO

import numpy as np

from AutoML2015.models.autosklearn import get_configuration_space
from AutoML2015.data.data_converter import predict_RAM_usage
from AutoML2015.util.automl_phase1 import get_dataset_list
from AutoSklearn.components.classification import _classifiers
from AutoSklearn.components.preprocessing import _preprocessors
from HPOlibConfigSpace.converters import pcs_parser

input_csv_file = "/home/feurerm/projects/openml/datasets/datasets_iteration002.csv"
output_dir = "/home/feurerm/ihome/projects/automl_competition_2015" \
             "/experiments/metalearning_for_phase_1"
smac = "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_08"


with open(input_csv_file) as fh:
    datasets = get_dataset_list(fh)


classifiers = _classifiers
preprocessors = [p for p in _preprocessors if p not in
                 ["imputation", "rescaling"]] + ["None"]
metrics = ["bac_metric", "auc_metric", "f1_metric", "pac_metric"]

print "#datasets", len(datasets)
print "#classifiers", len(classifiers)
print "#preprocessors", len(preprocessors)
print "#metrics", len(metrics)
print len(datasets) * len(classifiers) * len(preprocessors) * len(metrics)
print


# TODO Add possibility to restart by classifier or preprocessor...
for dataset, task_type in datasets:
    commands = []

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

    for classifier, preprocessor, metric in \
            product(classifiers, preprocessors, metrics):

        # Create configuration space, we can put everything in here because by
        # default it is not sparse and not multiclass and not multilabel
        try:
            configuration_space = get_configuration_space(
                {'task': task_type, 'is_sparse': sparse},
                include_classifiers=[classifier],
                include_preprocessors=[preprocessor])
        except ValueError as e:
            print e.message
            continue

        classifiers_ = ["extra_trees", "gradient_boosting",
                        "k_nearest_neighbors", "libsvm_svc", "random_forest"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]
        if classifier in classifiers_ and preprocessor in feature_learning_:
            continue

        if preprocessor != "None":
            pre = configuration_space.get_hyperparameter("preprocessor")
            item_number = pre.choices.index("None")
            del pre.choices[item_number]
            pre.default = preprocessor

        # Create the directories after creating the configuration space to
        # avoid creating empty directories!
        clf_pre_met_combination_dir = "%s-%s-%s" % \
                                      (classifier, preprocessor, metric)
        combination_directory = os.path.join(output_dir, clf_pre_met_combination_dir)
        try:
            os.mkdir(combination_directory)
        except:
            pass

        output_dir_ = os.path.join(combination_directory, str(dataset.id))
        if not os.path.exists(output_dir_):
            os.mkdir(output_dir_)
        else:
            continue

        with open(os.path.join(output_dir_, "params.pcs"), "w") as fh:
            cs = pcs_parser.write(configuration_space)
            fh.write(cs)

        config = StringIO()
        config.write("[HPOLIB]\n")
        config.write("dispatcher = runsolver_wrapper.py\n")
        config.write("function = python -m AutoML2015.wrapper.openml_wrapper "
                     "--dataset %d --metric %s --task_type %s --remove_categorical\n"
                     % (dataset.id, metric, task_type))
        config.write("number_of_jobs = 100\n")
        config.write("number_cv_folds = 10\n")
        config.write("runsolver_time_limit = 1800\n")
        config.write("memory_limit = 4000\n")
        config.write("result_on_terminate = 1.0\n")
        config.write("[SMAC]\n")
        config.write("runtime_limit = 172800\n")
        config.write("p = params.pcs\n")

        with open(os.path.join(output_dir_, "config.cfg"), "w") as fh:
            fh.write(config.getvalue())

        commands.append("HPOlib-run -o %s --cwd %s "
                        "--HPOLIB:temporary_output_directory "
                        "/tmp/${JOB_ID}.${SGE_TASK_ID}.aad_core.q"
                        % (smac, output_dir_))

    # TODO automatically create scripts, which can start every combination of
    #  classifiers with preproc and metric and so on...
    for seed in range(1000, 10001, 1000):
        with open(os.path.join(output_dir, "commands_seed-%d_did-%d.txt"
                % (seed, dataset.id)), "w") as fh:
            for command in commands:
                fh.write(command + " --seed %d --HPOLIB:optimizer_loglevel 10" % seed)
                fh.write("\n")
