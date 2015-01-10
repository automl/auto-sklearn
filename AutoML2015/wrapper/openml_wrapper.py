from functools import partial
import sys
import time

import numpy as np
from sklearn.preprocessing.label import LabelEncoder

from AutoSklearn.autosklearn import AutoSklearnClassifier

from HPOlibConfigSpace import configuration_space

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from openml.apiconnector import APIConnector

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import evaluate
from AutoML2015.util.split_data import split_data, get_CV_fold


class DataManagerDummy(DataManager):
    def __init__(self):
        pass


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

    e = None
    for i in range(60):
        e = None
        try:
            api = APIConnector(authenticate=False)
            dataset = api.get_cached_dataset(int(dataset))
            X, Y = dataset.get_pandas(target=dataset.default_target_attribute,
                                      include_row_id=False,
                                      include_ignore_attributes=False)
            break
        except Exception as e:
            time.sleep(1)
    if e is not None:
        print e
        sys.exit(1)

    feat_type = ['Numerical' if dtype == np.float64 else 'Categorical' for
                 dtype in X.dtypes]
    features = tuple([LabelEncoder().fit_transform(X.values[:,i]).reshape((-1, 1))
                      if feat_type[i] == 'Categorical' else
                      X.values[:,i].reshape((-1, 1)) for i in range(X.shape[1])])
    X_shape = X.shape
    X = np.hstack(features)
    assert X_shape == X.shape, (X_shape, X.shape)
    Y = LabelEncoder().fit_transform(Y.values).reshape((-1, ))
    assert len(Y) == len(X), (Y.shape, X.shapes)
    # Create train/test split:
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    print X_train
    print Y_train
    D = DataManagerDummy()
    D.data = {}
    D.data = {'X_train': X_train, 'Y_train': Y_train}
    D.info = {}
    D.info['task'] = task_type
    D.info['metric'] = metric
    D.info['is_sparse'] = 0
    D.info['has_missing'] = 0
    D.feat_type = feat_type
    D.perform1HotEncoding()

    def splitting_function(X, Y, fold, folds):
        train_indices, test_indices = \
            get_CV_fold(X, Y, fold=fold, folds=folds)
        print X.shape, Y.shape
        print train_indices.shape, max(train_indices),\
            test_indices.shape, max(test_indices)
        return  X[train_indices], X[test_indices], Y[train_indices], \
            Y[test_indices]
    splitting_function = partial(splitting_function, fold=fold, folds=folds)

    starttime = time.time()
    errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
        evaluate(D, configuration, with_predictions=True,
                 all_scoring_functions=True,
                 splitting_function=splitting_function)

    duration = time.time() - starttime

    err = errs[metric]
    additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
    additional_run_info += ";" + "duration: " + str(duration)
    return err, additional_run_info


if __name__ == "__main__":
    starttime = time.time()
    # Change a SMAC call into an HPOlib call, not yet needed!
    # if not "--params" in sys.argv:
    #    # Call from SMAC
    #    # Replace the SMAC seed by --params
    #    for i in range(len(sys.argv)):
    #        if sys.argv[i] == "2147483647" and sys.argv[i+1] == "-1":
    #            sys.argv[i+1] = "--params"

    args, params = parse_cli()

    result, additional_run_info = main(args, params)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
          ("SAT", abs(duration), result, -1, additional_run_info)
