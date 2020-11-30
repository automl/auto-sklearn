from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
import copy
import logging
import os
import sys
import unittest.mock

import arff
import joblib
import numpy as np
import pandas as pd

from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION
from autosklearn.metalearning.metafeatures import metafeatures
from autosklearn.smbo import _calculate_metafeatures, _calculate_metafeatures_encoded, \
    EXCLUDE_META_FEATURES_REGRESSION, EXCLUDE_META_FEATURES_CLASSIFICATION
from autosklearn.util.stopwatch import StopWatch

sys.path.append('.')
from update_metadata_util import load_task, classification_tasks, \
    regression_tasks

logger = logging.getLogger("03_calculate_metafeatures")


def calculate_metafeatures(task_id):
    X_train, y_train, X_test, y_test, cat, task_type, dataset_name = load_task(task_id)
    watch = StopWatch()

    if task_type == 'classification':
        if len(np.unique(y_train)) == 2:
            task_type = BINARY_CLASSIFICATION
        else:
            task_type = MULTICLASS_CLASSIFICATION
    else:
        task_type = REGRESSION

    _metafeatures_labels = _calculate_metafeatures(
        x_train=X_train, y_train=y_train, data_feat_type=cat,
        data_info_task=task_type, basename=dataset_name, logger=logger,
        watcher=watch,
    )

    _metafeatures_encoded_labels = _calculate_metafeatures_encoded(
        x_train=X_train, y_train=y_train, data_feat_type=cat,
        task=task_type, basename=dataset_name, logger=logger,
        watcher=watch,
    )

    mf = _metafeatures_labels
    mf.metafeature_values.update(
        _metafeatures_encoded_labels.metafeature_values)

    return mf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--working-directory", type=str, required=True)
    parser.add_argument("--memory-limit", type=int, default=3072)
    parser.add_argument("--test-mode", action='store_true')

    args = parser.parse_args()
    working_directory = args.working_directory
    memory_limit = args.memory_limit
    test_mode = args.test_mode

    for task_type in ('classification', 'regression'):
        output_directory = os.path.join(working_directory, 'metafeatures', task_type)
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

        memory = joblib.Memory(location='/tmp/joblib', verbose=10)
        cached_calculate_metafeatures = memory.cache(calculate_metafeatures)
        mfs = [
            cached_calculate_metafeatures(task_id)
            for task_id in producer()
        ]

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
        calculation_times = calculation_times.sort_index()
        with open(os.path.join(output_directory, "calculation_times.csv"),
                  "w") as fh:
            fh.write(calculation_times.to_csv())

        # Write all metafeatures in the aslib1.0 format
        metafeature_values = metafeature_values = pd.DataFrame(metafeature_values).transpose()
        metafeature_values = metafeature_values.sort_index()
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
                     for value in metafeature_values.loc[idx, :].values]
            data.append(line)
        arff_object['data'] = data

        with open(os.path.join(output_directory, "feature_values.arff"),
                  "w") as fh:
            arff.dump(arff_object, fh)

        # Feature steps and runtimes according to the aslib1.0 format
        feature_steps = defaultdict(list)
        metafeature_names = list()

        exclude_metafeatures = (
            EXCLUDE_META_FEATURES_CLASSIFICATION
            if task_type == 'classification' else EXCLUDE_META_FEATURES_REGRESSION
        )

        for metafeature_name in metafeatures.metafeatures.functions:

            if metafeature_name in exclude_metafeatures:
                continue

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
            for entry in description:
                fh.write("%s: %s\n" % (entry, description[entry]))
