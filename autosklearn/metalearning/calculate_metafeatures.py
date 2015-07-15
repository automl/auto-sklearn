from argparse import ArgumentParser
import resource
import os

import numpy as np

from autosklearn.data.data_manager import DataManager

import pyMetaLearn.metafeatures.metafeatures
import pyMetaLearn.metafeatures.metafeature


def calculate_metafeatures(dataset_name, dataset_dir, output_dir,
                           memory_limit=3000):
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1048576L, -1L))
    mf_filename = os.path.join(output_dir, "%s.arff" % dataset_name)

    D = DataManager(dataset_name, dataset_dir, verbose=False,
                    encode_labels=False)
    X = D.data["X_train"]
    y = D.data["Y_train"]

    if D.info['feat_type'].lower() == "categorical":
        categorical = [True] * int(float(D.info['feat_num']))
    elif D.info['feat_type'].lower() == "numerical":
        categorical = [False] * int(float(D.info['feat_num']))
    else:
        categorical = [True if c.lower() == "categorical" else False
                       for c in D.feat_type]

    _metafeatures_labels = pyMetaLearn.metafeatures.metafeatures.\
        calculate_all_metafeatures_with_labels(
        X, y, categorical, dataset_name)

    # Dump the metafeatures for safety in case that there is a crash later
    # http://stackoverflow.com/questions/14906962/python-double-free-error-for-huge-datasets
    _metafeatures_labels.dump(mf_filename)

    D.perform1HotEncoding()
    X = D.data["X_train"]
    y = D.data["Y_train"]
    categorical = [False] * X.shape[1]

    try:
        _metafeatures_encoded_labels = pyMetaLearn.metafeatures.metafeatures.\
            calculate_all_metafeatures_encoded_labels(
            X, y, categorical, dataset_name)
    except MemoryError as e:
        # During the conversion of the dataset (rescaling, etc...), it can
        # happen that we run out of memory.
        _metafeatures_encoded_labels = \
            pyMetaLearn.metafeatures.metafeature.DatasetMetafeatures(
                dataset_name, dict())
        for metafeature_name in \
                pyMetaLearn.metafeatures.metafeatures.npy_metafeatures:
            type_ = "HELPERFUNCTION" if metafeature_name not in \
                pyMetaLearn.metafeatures.metafeatures.metafeatures.functions \
                else "METAFEATURE"
            _metafeatures_encoded_labels.metafeature_values[metafeature_name] = \
                pyMetaLearn.metafeatures.metafeature.MetaFeatureValue(
                    metafeature_name, type_, 0, 0, np.NaN, np.NaN,
                    "Memory error during dataset scaling.")

    mf = _metafeatures_labels
    mf.metafeature_values.update(
        _metafeatures_encoded_labels.metafeature_values)

    mf.dump(mf_filename)

    return mf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--memory_limit", type=int, default=3000)
    args = parser.parse_args()

    mf = calculate_metafeatures(args.dataset_name, args.dataset_dir,
                                args.output_dir, args.memory_limit)