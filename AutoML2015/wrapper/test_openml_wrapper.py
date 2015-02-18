import time

from AutoSklearn.classification import AutoSklearnClassifier
from HPOlibConfigSpace import configuration_space

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from AutoML2015.util.split_data import split_data
import AutoML2015.wrapper.openml_wrapper
import AutoML2015.models.test_evaluate


def create_mock_test_data_manager(X, y, categorical, metric, task_type):
    """Create an object which looks like a data_manager to feed it to the
    evaluate module."""
    feat_type = ['Numerical' if not c else 'Categorical' for
                 c in categorical]
    # Create train/test split:
    X_train, X_test, Y_train, Y_test = split_data(X, y)
    D = AutoML2015.wrapper.openml_wrapper.DataManagerDummy()
    D.basename = ""
    D.data = {}
    D.data = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}
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

    openml_cache_directory = args.get("openml_cache_directory")
    dataset = args['dataset'].replace("/", "")
    metric = args['metric']
    task_type = args['task_type']
    if task_type not in ["multiclass.classification", "binary.classification",
                         "multilabel.classification"]:
        raise ValueError(task_type)

    cs = AutoSklearnClassifier.get_hyperparameter_search_space()
    configuration = configuration_space.Configuration(cs, **params)

    X, y, categorical = AutoML2015.wrapper.openml_wrapper.load_dataset(dataset, openml_cache_directory)
    if 'remove_categorical' in args:
        X, categorical = AutoML2015.wrapper.openml_wrapper.remove_categorical_features(X, categorical)
    D = create_mock_test_data_manager(X, y, categorical, metric, task_type)

    starttime = time.time()
    evaluator = AutoML2015.models.test_evaluate.Test_Evaluator(D, configuration,
                                                               with_predictions=True,
                                                               all_scoring_functions=True)
    evaluator.fit()
    #errs, Y_train_pred = evaluator.predict(train=True)
    errs, Y_test_pred = evaluator.predict(train=False)
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