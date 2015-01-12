from functools import partial
import sys
import time

import numpy as np

from AutoSklearn.autosklearn import AutoSklearnClassifier
from sklearn.preprocessing.label import LabelEncoder
from HPOlibConfigSpace import configuration_space

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from openml.apiconnector import APIConnector

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import Evaluator
from AutoML2015.util.split_data import split_data, get_CV_fold


class DataManagerDummy(DataManager):
    def __init__(self):
        pass


def load_dataset(dataset):
    """Load an OpenML datasets by its ID."""
    e = None
    for i in range(60):
        e = None
        try:
            api = APIConnector(authenticate=False)
            dataset = api.get_cached_dataset(int(dataset))
            X, y, categorical = dataset.get_pandas(
                target=dataset.default_target_attribute,
                include_row_id=False, include_ignore_attributes=False)
            break
        except Exception as e:
            time.sleep(1)
    if e is not None:
        print e
        sys.exit(1)

    # Perform label encoding, because otherwise the scoring functions won't work
    le = LabelEncoder()
    y = le.fit_transform(y.values)
    return X.values, y, categorical


def remove_categorical_features(X, categorical):
    """For AutoML phase 1 to obtain datasets which only have numerical features."""
    categorical = np.array(categorical)
    return X[:,~categorical], [False] * np.sum(~categorical)


def create_mock_data_manager(X, y, categorical, metric, task_type):
    """Create an object which looks like a data_manager to feed it to the
    evaluate module."""
    feat_type = ['Numerical' if not c else 'Categorical' for
                 c in categorical]
    # Create train/test split:
    X_train, X_test, Y_train, Y_test = split_data(X, y)
    D = DataManagerDummy()
    D.basename = ""
    D.data = {}
    D.data = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test}
    D.info = {}
    D.info['task'] = task_type
    D.info['metric'] = metric

    # TODO changes this for phase 2
    D.info['is_sparse'] = 0
    D.info['has_missing'] = 0
    D.feat_type = feat_type
    D.perform1HotEncoding()
    return D


def main(args, params):
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass

    dataset = args['dataset']
    metric = args['metric']
    task_type = args['task_type']
    if task_type not in ["multiclass.classification", "binary.classification",
                         "multilabel.classification"]:
        raise ValueError(task_type)
    fold = int(args['fold'])
    folds = int(args['folds'])

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    X, y, categorical = load_dataset(dataset)
    if 'remove_categorical' in args:
        X, categorical = remove_categorical_features(X, categorical)
    D = create_mock_data_manager(X, y, categorical, metric, task_type)

    def splitting_function(X, Y, fold, folds):
        train_indices, test_indices = \
            get_CV_fold(X, Y, fold=fold, folds=folds)
        return X[train_indices], X[test_indices], Y[train_indices], \
            Y[test_indices]
    splitting_function = partial(splitting_function, fold=fold, folds=folds)

    starttime = time.time()
    evaluator = Evaluator(D, configuration, with_predictions=True,
                          all_scoring_functions=True,
                          splitting_function=splitting_function)
    evaluator.fit()
    errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = evaluator.predict()
    duration = time.time() - starttime

    err = errs[metric]
    additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
    additional_run_info += ";" + "duration: " + str(duration)
    return err, additional_run_info


if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()

    result, additional_run_info = main(args, params)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
          ("SAT", abs(duration), result, -1, additional_run_info)
