from argparse import ArgumentParser
import resource
import os

import numpy as np

import autosklearn.data.data_manager_factory as data_manager_factory

import autosklearn.metalearning.metafeatures.metafeatures
import autosklearn.metalearning.metafeatures.metafeature


def calculate_metafeatures(D, output_dir, memory_limit=3072):
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1048576, -1))
    mf_filename = os.path.join(output_dir, "%s.arff" % D.name)

    X = D.data["X_train"]
    y = D.data["Y_train"]

    if D.info['feat_type'].lower() == "categorical":
        categorical = [True] * int(float(D.info['feat_num']))
    elif D.info['feat_type'].lower() == "numerical":
        categorical = [False] * int(float(D.info['feat_num']))
    else:
        categorical = [True if c.lower() == "categorical" else False
                       for c in D.feat_type]

    _metafeatures_labels = autosklearn.metalearning.metafeatures.metafeatures. \
        calculate_all_metafeatures_with_labels(
        X, y, categorical, D.name)

    # Dump the metafeatures for safety in case that there is a crash later
    # http://stackoverflow.com/questions/14906962/python-double-free-error-for-huge-datasets
    _metafeatures_labels.dump(mf_filename)

    D.perform1HotEncoding()
    X = D.data["X_train"]
    y = D.data["Y_train"]
    categorical = [False] * X.shape[1]

    try:
        _metafeatures_encoded_labels = autosklearn.metalearning.metafeatures. \
            metafeatures.calculate_all_metafeatures_encoded_labels(
            X, y, categorical, D.name)
    except MemoryError as e:
        # During the conversion of the dataset (rescaling, etc...), it can
        # happen that we run out of memory.
        _metafeatures_encoded_labels = \
            autosklearn.metalearning.metafeatures.metafeature.DatasetMetafeatures(
                D.name, dict())
        for metafeature_name in \
                autosklearn.metalearning.metafeatures.metafeatures.npy_metafeatures:
            type_ = "HELPERFUNCTION" if metafeature_name not in \
                                        autosklearn.metalearning.metafeatures.metafeatures.metafeatures.functions \
                else "METAFEATURE"
            _metafeatures_encoded_labels.metafeature_values[metafeature_name] = \
                autosklearn.metalearning.metafeatures.metafeature.MetaFeatureValue(
                    metafeature_name, type_, 0, 0, np.NaN, np.NaN,
                    "Memory error during dataset scaling.")

    mf = _metafeatures_labels
    mf.metafeature_values.update(
        _metafeatures_encoded_labels.metafeature_values)

    mf.dump(mf_filename)

    return mf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--memory-limit", type=int, default=3072)
    parser = data_manager_factory.populate_argparse_with_data_options(parser)
    args = parser.parse_args()

    D = data_manager_factory.get_data_manager(args, encode_labels=False)
    mf = calculate_metafeatures(D, args.output_dir, args.memory_limit)