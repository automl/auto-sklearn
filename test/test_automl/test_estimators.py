import glob
import os
import pickle
import re
import sys
import unittest
import unittest.mock
import pytest

import joblib
from joblib import cpu_count
import numpy as np
import numpy.ma as npma
import pandas as pd
import sklearn
import sklearn.dummy
import sklearn.datasets

import autosklearn.pipeline.util as putil
from autosklearn.ensemble_builder import MODEL_FN_RE
import autosklearn.estimators  # noqa F401
from autosklearn.estimators import AutoSklearnEstimator
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import accuracy, f1_macro, mean_squared_error, r2
from autosklearn.automl import AutoMLClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.smbo import get_smac_object

sys.path.append(os.path.dirname(__file__))
from automl_utils import print_debug_information, count_succeses  # noqa (E402: module level import not at top of file)


def test_fit_n_jobs(tmp_dir, output_dir):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')

    # test parallel Classifier to predict classes, not only indices
    Y_train += 1
    Y_test += 1

    class get_smac_object_wrapper:

        def __call__(self, *args, **kwargs):
            self.n_jobs = kwargs['n_jobs']
            smac = get_smac_object(*args, **kwargs)
            self.dask_n_jobs = smac.solver.tae_runner.n_workers
            self.dask_client_n_jobs = len(
                smac.solver.tae_runner.client.scheduler_info()['workers']
            )
            return smac
    get_smac_object_wrapper_instance = get_smac_object_wrapper()

    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        output_folder=output_dir,
        tmp_folder=tmp_dir,
        seed=1,
        initial_configurations_via_metalearning=0,
        ensemble_size=5,
        n_jobs=2,
        include_estimators=['sgd'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_wrapper_instance,
        max_models_on_disc=None,
    )
    automl.fit(X_train, Y_train)

    # Test that the argument is correctly passed to SMAC
    assert getattr(get_smac_object_wrapper_instance, 'n_jobs') == 2
    assert getattr(get_smac_object_wrapper_instance, 'dask_n_jobs') == 2
    assert getattr(get_smac_object_wrapper_instance, 'dask_client_n_jobs') == 2

    available_num_runs = set()
    for run_key, run_value in automl.automl_.runhistory_.data.items():
        if run_value.additional_info is not None and 'num_run' in run_value.additional_info:
            available_num_runs.add(run_value.additional_info['num_run'])
    available_predictions = set()
    predictions = glob.glob(
        os.path.join(automl.automl_._backend.get_runs_directory(), '*', 'predictions_ensemble*.npy')
    )
    seeds = set()
    for prediction in predictions:
        prediction = os.path.split(prediction)[1]
        match = re.match(MODEL_FN_RE, prediction.replace("predictions_ensemble", ""))
        if match:
            num_run = int(match.group(2))
            available_predictions.add(num_run)
            seed = int(match.group(1))
            seeds.add(seed)

    # Remove the dummy prediction, it is not part of the runhistory
    available_predictions.remove(1)
    assert available_num_runs.issubset(available_predictions)

    assert len(seeds) == 1

    ensemble_dir = automl.automl_._backend.get_ensemble_dir()
    ensembles = os.listdir(ensemble_dir)

    seeds = set()
    for ensemble_file in ensembles:
        seeds.add(int(ensemble_file.split('.')[0].split('_')[0]))
    assert len(seeds) == 1

    assert count_succeses(automl.cv_results_) > 0
    # For travis-ci it is important that the client no longer exists
    assert automl.automl_._dask_client is None


def test_feat_type_wrong_arguments():
    cls = AutoSklearnClassifier(ensemble_size=0)
    X = np.zeros((100, 100))
    y = np.zeros((100, ))

    expected_msg = r".*Array feat_type does not have same number of "
    "variables as X has features. 1 vs 100.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True])

    expected_msg = r".*Array feat_type must only contain strings.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True]*100)

    expected_msg = r".*Only `Categorical` and `Numerical` are"
    "valid feature types, you passed `Car`.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=['Car']*100)


# Mock AutoSklearnEstimator.fit so the test doesn't actually run fit().
@unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.fit')
def test_type_of_target(mock_estimator):
    # Test that classifier raises error for illegal target types.
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  ])
    # Possible target types
    y_binary = np.array([0, 0, 1, 1])
    y_continuous = np.array([0.1, 1.3, 2.1, 4.0])
    y_multiclass = np.array([0, 1, 2, 0])
    y_multilabel = np.array([[0, 1],
                             [1, 1],
                             [1, 0],
                             [0, 0],
                             ])
    y_multiclass_multioutput = np.array([[0, 1],
                                         [1, 3],
                                         [2, 2],
                                         [5, 3],
                                         ])
    y_continuous_multioutput = np.array([[0.1, 1.5],
                                         [1.2, 3.5],
                                         [2.7, 2.7],
                                         [5.5, 3.9],
                                         ])

    cls = AutoSklearnClassifier(ensemble_size=0)
    # Illegal target types for classification: continuous,
    # multiclass-multioutput, continuous-multioutput.
    expected_msg = r".*Classification with data of type"
    " multiclass-multioutput is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y_multiclass_multioutput)

    expected_msg = r".*Classification with data of type"
    " continuous is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y_continuous)

    expected_msg = r".*Classification with data of type"
    " continuous-multioutput is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y_continuous_multioutput)

    # Legal target types for classification: binary, multiclass,
    # multilabel-indicator.
    try:
        cls.fit(X, y_binary)
    except ValueError:
        pytest.fail("cls.fit() raised ValueError while fitting "
                    "binary targets")

    try:
        cls.fit(X, y_multiclass)
    except ValueError:
        pytest.fail("cls.fit() raised ValueError while fitting "
                    "multiclass targets")

    try:
        cls.fit(X, y_multilabel)
    except ValueError:
        pytest.fail("cls.fit() raised ValueError while fitting "
                    "multilabel-indicator targets")

    # Test that regressor raises error for illegal target types.
    reg = AutoSklearnRegressor(ensemble_size=0)
    # Illegal target types for regression: multilabel-indicator
    # multiclass-multioutput
    expected_msg = r".*Regression with data of type"
    " multilabel-indicator is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        reg.fit(X=X, y=y_multilabel,)

    expected_msg = r".*Regression with data of type"
    " multiclass-multioutput is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        reg.fit(X=X, y=y_multiclass_multioutput,)

    # Legal target types: continuous, multiclass,
    # continuous-multioutput,
    # binary
    try:
        reg.fit(X, y_continuous)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting "
                    "continuous targets")

    try:
        reg.fit(X, y_multiclass)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting "
                    "multiclass targets")

    try:
        reg.fit(X, y_continuous_multioutput)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting "
                    "continuous_multioutput targets")

    try:
        reg.fit(X, y_binary)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting "
                    "binary targets")


def test_cv_results(tmp_dir, output_dir):
    # TODO restructure and actually use real SMAC output from a long run
    # to do this unittest!
    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

    cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                per_run_time_limit=5,
                                tmp_folder=tmp_dir,
                                output_folder=output_dir,
                                seed=1,
                                initial_configurations_via_metalearning=0,
                                ensemble_size=0,
                                scoring_functions=[autosklearn.metrics.precision,
                                                   autosklearn.metrics.roc_auc])
    cls.fit(X_train, Y_train)
    cv_results = cls.cv_results_
    assert isinstance(cv_results, dict), type(cv_results)
    assert isinstance(cv_results['mean_test_score'], np.ndarray), type(
        cv_results['mean_test_score'])
    assert isinstance(cv_results['mean_fit_time'], np.ndarray), type(
        cv_results['mean_fit_time']
    )
    assert isinstance(cv_results['params'], list), type(cv_results['params'])
    assert isinstance(cv_results['rank_test_scores'], np.ndarray), type(
        cv_results['rank_test_scores']
    )
    assert isinstance(cv_results['metric_precision'], npma.MaskedArray), type(
        cv_results['metric_precision']
    )
    assert isinstance(cv_results['metric_roc_auc'], npma.MaskedArray), type(
        cv_results['metric_roc_auc']
    )
    cv_result_items = [isinstance(val, npma.MaskedArray) for key, val in
                       cv_results.items() if key.startswith('param_')]
    assert all(cv_result_items), cv_results.items()


@unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.build_automl')
def test_fit_n_jobs_negative(build_automl_patch):
    n_cores = cpu_count()
    cls = AutoSklearnEstimator(n_jobs=-1, ensemble_size=0)
    cls.fit()
    assert cls._n_jobs == n_cores


def test_get_number_of_available_cores():
    n_cores = cpu_count()
    assert n_cores >= 1, n_cores


@unittest.mock.patch('autosklearn.automl.AutoML.predict')
def test_multiclass_prediction(predict_mock, backend, dask_client):
    predicted_probabilities = [[0, 0, 0.99], [0, 0.99, 0], [0.99, 0, 0],
                               [0, 0.99, 0], [0, 0, 0.99]]
    predicted_indexes = [2, 1, 0, 1, 2]
    expected_result = ['c', 'b', 'a', 'b', 'c']

    predict_mock.return_value = np.array(predicted_probabilities)

    classifier = AutoMLClassifier(
        time_left_for_this_task=1,
        per_run_time_limit=1,
        backend=backend,
        dask_client=dask_client,
    )
    classifier.InputValidator.validate_target(
        pd.DataFrame(expected_result, dtype='category'),
        is_classification=True,
    )

    actual_result = classifier.predict([None] * len(predicted_indexes))

    np.testing.assert_array_equal(expected_result, actual_result)


@unittest.mock.patch('autosklearn.automl.AutoML.predict')
def test_multilabel_prediction(predict_mock, backend, dask_client):
    predicted_probabilities = [[0.99, 0],
                               [0.99, 0],
                               [0, 0.99],
                               [0.99, 0.99],
                               [0.99, 0.99]]
    predicted_indexes = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [1, 1]])
    expected_result = np.array([[2, 13], [2, 13], [1, 17], [2, 17], [2, 17]])

    predict_mock.return_value = np.array(predicted_probabilities)

    classifier = AutoMLClassifier(
        time_left_for_this_task=1,
        per_run_time_limit=1,
        backend=backend,
        dask_client=dask_client,
    )
    classifier.InputValidator.validate_target(
        pd.DataFrame(expected_result, dtype='int64'),
        is_classification=True,
    )

    actual_result = classifier.predict([None] * len(predicted_indexes))

    np.testing.assert_array_equal(expected_result, actual_result)


def test_can_pickle_classifier(tmp_dir, output_dir, dask_client):
    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                   per_run_time_limit=5,
                                   tmp_folder=tmp_dir,
                                   dask_client=dask_client,
                                   output_folder=output_dir)
    automl.fit(X_train, Y_train)

    initial_predictions = automl.predict(X_test)
    initial_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                      initial_predictions)
    assert initial_accuracy >= 0.75
    assert count_succeses(automl.cv_results_) > 0

    # Test pickle
    dump_file = os.path.join(output_dir, 'automl.dump.pkl')

    with open(dump_file, 'wb') as f:
        pickle.dump(automl, f)

    with open(dump_file, 'rb') as f:
        restored_automl = pickle.load(f)

    restored_predictions = restored_automl.predict(X_test)
    restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                       restored_predictions)
    assert restored_accuracy >= 0.75
    assert initial_accuracy == restored_accuracy

    # Test joblib
    dump_file = os.path.join(output_dir, 'automl.dump.joblib')

    joblib.dump(automl, dump_file)

    restored_automl = joblib.load(dump_file)

    restored_predictions = restored_automl.predict(X_test)
    restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                       restored_predictions)
    assert restored_accuracy >= 0.75
    assert initial_accuracy == restored_accuracy


def test_multilabel(tmp_dir, output_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset(
        'iris', make_multilabel=True)
    automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                   per_run_time_limit=5,
                                   tmp_folder=tmp_dir,
                                   dask_client=dask_client,
                                   output_folder=output_dir)

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (50, 3), print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0,  print_debug_information(automl)

    score = f1_macro(Y_test, predictions)
    assert score >= 0.9, print_debug_information(automl)

    probs = automl.predict_proba(X_train)
    assert np.mean(probs) == pytest.approx(0.33, rel=1e-1)


def test_binary(tmp_dir, output_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset(
        'iris', make_binary=True)
    automl = AutoSklearnClassifier(time_left_for_this_task=40,
                                   per_run_time_limit=10,
                                   tmp_folder=tmp_dir,
                                   dask_client=dask_client,
                                   output_folder=output_dir)

    automl.fit(X_train, Y_train, X_test=X_test, y_test=Y_test,
               dataset_name='binary_test_dataset')

    predictions = automl.predict(X_test)
    assert predictions.shape == (50, ), print_debug_information(automl)

    score = accuracy(Y_test, predictions)
    assert score > 0.9, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)

    output_files = glob.glob(os.path.join(output_dir, 'binary_test_dataset_test_*.predict'))
    assert len(output_files) > 0, (output_files, print_debug_information(automl))


def test_classification_pandas_support(tmp_dir, output_dir, dask_client):

    X, y = sklearn.datasets.fetch_openml(
        data_id=2,  # cat/num dataset
        return_X_y=True,
        as_frame=True,
    )

    # Drop NAN!!
    X = X.dropna('columns')

    # This test only make sense if input is dataframe
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        exclude_estimators=['libsvm_svc'],
        dask_client=dask_client,
        seed=5,
        tmp_folder=tmp_dir,
        output_folder=output_dir,
    )

    automl.fit(X, y)

    # Make sure that at least better than random.
    # We use same X_train==X_test to test code quality
    assert automl.score(X, y) > 0.555, print_debug_information(automl)

    automl.refit(X, y)

    # Make sure that at least better than random.
    # accuracy in sklearn needs valid data
    # It should be 0.555 as the dataset is unbalanced.
    y = automl.automl_.InputValidator.encode_target(y)
    prediction = automl.automl_.InputValidator.encode_target(automl.predict(X))
    assert accuracy(y, prediction) > 0.555
    assert count_succeses(automl.cv_results_) > 0


def test_regression(tmp_dir, output_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('boston')
    automl = AutoSklearnRegressor(time_left_for_this_task=30,
                                  per_run_time_limit=5,
                                  tmp_folder=tmp_dir,
                                  dask_client=dask_client,
                                  output_folder=output_dir)

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (356,)
    score = mean_squared_error(Y_test, predictions)

    # On average np.sqrt(30) away from the target -> ~5.5 on average
    # Results with select rates drops avg score to a range of -32.40 to -37, on 30 seconds
    # constraint. With more time_left_for_this_task this is no longer an issue
    assert score >= -37, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0


def test_cv_regression(tmp_dir, output_dir, dask_client):
    """
    Makes sure that when using a cv strategy, we are able to fit
    a regressor
    """

    X_train, Y_train, X_test, Y_test = putil.get_dataset('boston', train_size_maximum=300)
    automl = AutoSklearnRegressor(time_left_for_this_task=60,
                                  per_run_time_limit=10,
                                  resampling_strategy='cv',
                                  tmp_folder=tmp_dir,
                                  dask_client=dask_client,
                                  output_folder=output_dir)

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (206,)
    score = r2(Y_test, predictions)
    assert score >= 0.1, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)


def test_regression_pandas_support(tmp_dir, output_dir, dask_client):

    X, y = sklearn.datasets.fetch_openml(
        data_id=41514,  # diabetes
        return_X_y=True,
        as_frame=True,
    )
    # This test only make sense if input is dataframe
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    automl = AutoSklearnRegressor(
        time_left_for_this_task=40,
        per_run_time_limit=5,
        dask_client=dask_client,
        tmp_folder=tmp_dir,
        output_folder=output_dir,
    )

    # Make sure we error out because y is not encoded
    automl.fit(X, y)

    # Make sure that at least better than random.
    # We use same X_train==X_test to test code quality
    assert automl.score(X, y) >= 0.5, print_debug_information(automl)

    automl.refit(X, y)

    # Make sure that at least better than random.
    assert r2(y, automl.predict(X)) > 0.5, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)


# Currently this class only tests that the methods of AutoSklearnClassifier
# which should return self actually return self.
def test_autosklearn_classification_methods_returns_self(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset('iris')
    automl = AutoSklearnClassifier(time_left_for_this_task=60,
                                   per_run_time_limit=10,
                                   ensemble_size=0,
                                   dask_client=dask_client,
                                   exclude_preprocessors=['fast_ica'])

    automl_fitted = automl.fit(X_train, y_train)
    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())
    assert automl is automl_refitted


# Currently this class only tests that the methods of AutoSklearnRegressor
# that should return self actually return self.
def test_autosklearn_regression_methods_returns_self(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset('boston')
    automl = AutoSklearnRegressor(time_left_for_this_task=30,
                                  per_run_time_limit=5,
                                  dask_client=dask_client,
                                  ensemble_size=0)

    automl_fitted = automl.fit(X_train, y_train)
    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())
    assert automl is automl_refitted


# Currently this class only tests that the methods of AutoSklearnClassifier
# which should return self actually return self.
def test_autosklearn2_classification_methods_returns_self(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset('iris')
    automl = AutoSklearn2Classifier(time_left_for_this_task=60, ensemble_size=0,
                                    dask_client=dask_client)

    automl_fitted = automl.fit(X_train, y_train)
    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())
    assert automl is automl_refitted

    predictions = automl_fitted.predict(X_test)
    assert sklearn.metrics.accuracy_score(
        y_test, predictions
    ) >= 2 / 3, print_debug_information(automl)

    pickle.dumps(automl_fitted)
