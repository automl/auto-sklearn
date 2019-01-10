from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
import copy
import os
import sys
import time

import arff
import joblib
import numpy as np
import pandas as pd
import pynisher
import scipy.sparse

from autosklearn.data.abstract_data_manager import perform_one_hot_encoding
from autosklearn.metalearning.metafeatures import metafeatures, metafeature

sys.path.append('.')
from update_metadata_util import load_task, classification_tasks, \
    regression_tasks


def calculate_metafeatures(task_id):
    print(task_id)
    X_train, y_train, X_test, y_test, cat = load_task(task_id)
    categorical = [True if 'categorical' == c else False for c in cat]

    _metafeatures_labels = metafeatures.calculate_all_metafeatures_with_labels(
        X_train, y_train, [False] * X_train.shape[1], task_id)

    X_train, sparse = perform_one_hot_encoding(scipy.sparse.issparse(X_train),
                                               categorical, [X_train])
    X_train = X_train[0]
    categorical = [False] * X_train.shape[1]

    start_time = time.time()
    obj = pynisher.enforce_limits(mem_in_mb=3072)(
        metafeatures.calculate_all_metafeatures_encoded_labels)
    _metafeatures_encoded_labels = obj(X_train, y_train,
                                       categorical, task_id)
    end_time = time.time()

    if obj.exit_status == pynisher.MemorylimitException:
        # During the conversion of the dataset (rescaling, etc...), it can
        # happen that we run out of memory.
        _metafeatures_encoded_labels = \
            metafeature.DatasetMetafeatures(task_id, dict())

        metafeature_calculation_time = (end_time - start_time) / \
                                       len(metafeatures.npy_metafeatures)

        for metafeature_name in metafeatures.npy_metafeatures:
            type_ = "HELPERFUNCTION" if metafeature_name not in \
                                        metafeatures.metafeatures.functions \
                else "METAFEATURE"
            _metafeatures_encoded_labels.metafeature_values[metafeature_name] = \
                metafeature.MetaFeatureValue(metafeature_name, type_, 0, 0,
                                             np.NaN, metafeature_calculation_time,
                                             "Memory error during dataset scaling.")

    mf = _metafeatures_labels
    mf.metafeature_values.update(
        _metafeatures_encoded_labels.metafeature_values)

    return mf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--working-directory", type=str, required=True)
    parser.add_argument("--task-type", required=True,
                        choices=['classification', 'regression'])
    parser.add_argument("--memory-limit", type=int, default=3072)
    parser.add_argument("--n-jobs",
                        help="Compute metafeatures in parallel if possible.",
                        type=int, default=1)
    parser.add_argument("--test-mode", type=bool, default=False)

    args = parser.parse_args()
    working_directory = args.working_directory
    task_type = args.task_type
    memory_limit = args.memory_limit
    n_jobs = args.n_jobs
    test_mode = args.test_mode

    output_directory = os.path.join(working_directory, 'metafeatures')
    try:
        os.makedirs(output_directory)
    except:
        pass

    all_metafeatures = {}

    if task_type == 'classification':
        tasks = classification_tasks
    else:
        tasks = regression_tasks

    if test_mode:
        tasks = [tasks[0]]

    tasks = copy.deepcopy(tasks)
    np.random.shuffle(tasks)

    def producer():
        for task_id in tasks:
            yield task_id

    memory = joblib.Memory(cachedir='/tmp/joblib', verbose=10)
    cached_calculate_metafeatures = memory.cache(calculate_metafeatures)
    mfs = joblib.Parallel(n_jobs=args.n_jobs) \
        (joblib.delayed(cached_calculate_metafeatures)(task_id)
         for task_id in producer())

    for mf in mfs:
        if mf is not None:
            all_metafeatures[mf.dataset_name] = mf

    # Write the calculation times as a csv file to disc (can be viewed in
    # LibreOffice calc afterwards)
    calculation_times = defaultdict(dict)
    metafeature_values = defaultdict(dict)
    helperfunction_values = defaultdict(dict)

    for i, task_id in enumerate(all_metafeatures):
        calculation_times[task_id] = dict()
        for metafeature_name in sorted(
                all_metafeatures[task_id].metafeature_values):
            metafeature_value = all_metafeatures[task_id].metafeature_values[
                metafeature_name]
            calculation_times[task_id][metafeature_name] = \
                metafeature_value.time
            if metafeature_value.type_ == "HELPERFUNCTION":
                helperfunction_values[task_id][metafeature_name] = \
                    metafeature_value.value
            else:
                metafeature_values[task_id][metafeature_name] = \
                    metafeature_value.value

    calculation_times = pd.DataFrame(calculation_times).transpose()
    with open(os.path.join(output_directory, "calculation_times.csv"),
              "w") as fh:
        fh.write(calculation_times.to_csv())

    # Write all metafeatures in the aslib1.0 format
    metafeature_values = pd.DataFrame(metafeature_values).transpose()
    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC')] + \
                                [('%s' % name, 'NUMERIC') for name in
                                 metafeature_values.columns]
    arff_object['relation'] = "FEATURE_VALUES"
    arff_object['description'] = ""

    data = []
    for idx in metafeature_values.index:
        line = [idx, 1]
        line += [value if np.isfinite(value) else None
                 for value in metafeature_values.ix[idx, :].values]
        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(output_directory, "feature_values.arff"),
              "w") as fh:
        arff.dump(arff_object, fh)

    # Feature steps and runtimes according to the aslib1.0 format
    feature_steps = defaultdict(list)
    metafeature_names = list()
    for metafeature_name in metafeatures.metafeatures.functions:
        dependency = metafeatures.metafeatures.get_dependency(metafeature_name)
        if dependency is not None:
            feature_steps[dependency].append(metafeature_name)
        feature_steps[metafeature_name].append(metafeature_name)

        metafeature_names.append(metafeature_name)

    # Write the feature runstatus in the aslib1.0 format
    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC')] + \
                                [('%s' % name,
                                  ['ok', 'timeout', 'memout', 'presolved',
                                   'crash', 'other'])
                                 for name in feature_steps]
    arff_object['relation'] = "FEATURE_RUNSTATUS"
    arff_object['description'] = ""

    data = []
    for idx in metafeature_values.index:
        line = [idx, 1]
        for feature_step in feature_steps:
            if feature_step in helperfunction_values[idx]:
                line.append('ok' if helperfunction_values[feature_step] is not \
                                    None else 'other')
            elif feature_step in metafeature_values.loc[idx]:
                line.append('ok' if np.isfinite(metafeature_values.loc[idx][
                                                    feature_step]) else 'other')
            else:
                line.append('other')

        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(output_directory, "feature_runstatus.arff"),
              "w") as fh:
        arff.dump(arff_object, fh)

    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC')] + \
                                [('%s' % feature_step, 'NUMERIC') for
                                 feature_step in feature_steps]
    arff_object['relation'] = "FEATURE_COSTS"
    arff_object['description'] = ""

    data = []
    for instance_id in calculation_times.index:
        calculation_times_per_group = dict()
        line = [instance_id, 1]
        for feature_step in feature_steps:
            time_ = 0.0
            for feature in feature_steps[feature_step]:
                time_ += calculation_times[feature][instance_id]
            if not np.isfinite(time_):
                raise ValueError("Feature cost %s for instance %s and feature "
                                 "step %s not finite" % (time_, instance_id, feature))
            line.append(time_)
        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(output_directory, "feature_costs.arff"),
              "w") as fh:
        arff.dump(arff_object, fh)

    # Write the features part of the description.txt to a file
    description = OrderedDict()
    description['features_cutoff_time'] = '3600'
    description['features_cutoff_memory'] = args.memory_limit
    description['number_of_feature_steps'] = str(len(feature_steps))

    for feature_step in feature_steps:
        description['feature_step %s' % feature_step] = \
            ", ".join(feature_steps[feature_step])
    description['features_deterministic'] = ", ".join([
                                                          metafeature_name for
                                                          metafeature_name in
                                                          metafeature_names])
    description['features_stochastic'] = ''
    description['default_steps'] = ", ".join(feature_steps)

    with open(os.path.join(output_directory,
                           "description.features.txt"), "w") as fh:
        for task_id in description:
            fh.write("%s: %s\n" % (task_id, description[task_id]))
