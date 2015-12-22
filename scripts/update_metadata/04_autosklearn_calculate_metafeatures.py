from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
import functools
import os
import subprocess

import arff
import joblib
import numpy as np
import pandas as pd

from autosklearn.metalearning.metafeatures import metafeatures
from autosklearn.metalearning.metafeatures.metafeature import DatasetMetafeatures


def calculate_metafeatures(dataset, output_dir, cache_only):
    dataset_dir, dataset_name = dataset
    dataset = os.path.join(dataset[0], dataset[1])
    mf_filename = os.path.join(output_dir, "%s.arff" % dataset_name)

    if os.path.exists(mf_filename):
        with open(mf_filename) as fh:
            mf = DatasetMetafeatures.load(fh)

    elif cache_only is False:
        print(dataset_name)

        call = "python -m autosklearn.metalearning.calculate_metafeatures " \
               "--data-format automl-competition-format --dataset %s " \
               "--output-dir %s" % (dataset, output_dir)
        print(call)
        retval = subprocess.call(call, shell=True)

        if retval != 0:
            return None

        with open(mf_filename) as fh:
            mf = DatasetMetafeatures.load(fh)

    else:
        return None

    return mf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets",
                        help="Text file with one dataset per line.")
    parser.add_argument("output_directory", help="Output directory where "
                                                 "metadata will be written.")
    parser.add_argument("--memory-limit", type=int, default=3072)
    parser.add_argument("--cache-only", help="Only use metafeatures which are "
                                             "already on the hard_drive.",
                        type=bool, default=False)
    parser.add_argument("--n-jobs", help="Compute metafeatures in parallel if possible.",
                        type=int, default=1)

    args = parser.parse_args()
    try:
        os.makedirs(args.output_directory)
    except:
        pass

    all_metafeatures = {}

    datasets = []
    datasets_csv = args.datasets
    with open(datasets_csv) as fh:
        for line in fh:
            line = line.replace("\n", "")
            if line.endswith("/"):
                line = line[:-1]
            dataset_dir, dataset_name = os.path.split(line)
            dataset_dir = os.path.abspath(dataset_dir)
            datasets.append((dataset_dir, dataset_name))

    def producer():
        for dataset_dir, dataset_name in datasets:
            yield (dataset_dir, dataset_name)

    cmf = functools.partial(calculate_metafeatures,
                            output_dir=args.output_directory,
                            cache_only=args.cache_only)
    mfs = joblib.Parallel(n_jobs=args.n_jobs)\
        (joblib.delayed(cmf)(dataset) for dataset in producer())

    for mf in mfs:
        if mf is not None:
            all_metafeatures[mf.dataset_name] = mf

    # Write the calculation times as a csv file to disc (can be viewed in
    # LibreOffice calc afterwards)
    calculation_times = defaultdict(dict)
    metafeature_values = defaultdict(dict)
    helperfunction_values = defaultdict(dict)
    for i, key in enumerate(all_metafeatures):
        calculation_times[key] = dict()
        for metafeature_value in sorted(all_metafeatures[key].metafeature_values):
            calculation_times[key][metafeature_value.name] = \
                metafeature_value.time
            if metafeature_value.type_ == "HELPERFUNCTION":
                helperfunction_values[key][metafeature_value.name] = \
                    metafeature_value.value
            else:
                metafeature_values[key][metafeature_value.name] = \
                    metafeature_value.value


    calculation_times = pd.DataFrame(calculation_times).transpose()
    with open(os.path.join(args.output_directory, "calculation_times.csv"),
              "w") as fh:
        fh.write(calculation_times.to_csv())

    # Write all metafeatures in the aslib1.0 format
    metafeature_values = pd.DataFrame(metafeature_values).transpose()
    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC')] + \
        [('%s' % name, 'NUMERIC') for name in metafeature_values.columns]
    arff_object['relation'] = "FEATURE_VALUES"
    arff_object['description'] = ""

    data = []
    for idx in metafeature_values.index:
        line = [idx, 1]
        line += [value if np.isfinite(value) else None
                 for value in metafeature_values.ix[idx,:].values]
        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(args.output_directory, "feature_values.arff"),
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
        [('%s' % name, ['ok', 'timeout', 'memout', 'presolved', 'crash' ,
                        'other'])
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
            else:
                line.append('ok' if np.isfinite(metafeature_values.loc[idx][
                            feature_step]) else 'other')

        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(args.output_directory, "feature_runstatus.arff"),
            "w") as fh:
        arff.dump(arff_object, fh)

    arff_object = dict()
    arff_object['attributes'] = [('instance_id', 'STRING'),
                                 ('repetition', 'NUMERIC')] + \
        [('%s' % feature_step, 'NUMERIC') for feature_step in feature_steps]
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
                raise ValueError("Feature cost for instance %s and feature "
                                 "step %s not finite" % (instance_id, feature))
            line.append(time_)
        data.append(line)
    arff_object['data'] = data

    with open(os.path.join(args.output_directory, "feature_costs.arff"),
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
        metafeature_name for metafeature_name in metafeature_names])
    description['features_stochastic'] = ''
    description['default_steps'] = ", ".join(feature_steps)

    with open(os.path.join(args.output_directory,
                           "description.features.txt"), "w") as fh:
        for key in description:
            fh.write("%s: %s\n" % (key, description[key]))