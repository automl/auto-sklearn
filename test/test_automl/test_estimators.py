import copy
import glob
import importlib
import os
import inspect
import pickle
import re
import sys
import tempfile
import unittest
import unittest.mock
import pytest

from ConfigSpace import Configuration
import joblib
from joblib import cpu_count
import numpy as np
import numpy.ma as npma
import pandas as pd
import sklearn
import sklearn.dummy
import sklearn.datasets
from sklearn.base import clone
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from smac.tae import StatusType

from autosklearn.data.validation import InputValidator
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

    # Every Auto-Sklearn estimator has a backend, that allows a single
    # call to fit
    X = np.zeros((100, 100))
    y = np.zeros((100, ))

    cls = AutoSklearnClassifier(ensemble_size=0)
    expected_msg = r".*Array feat_type does not have same number of "
    "variables as X has features. 1 vs 100.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True])

    cls = AutoSklearnClassifier(ensemble_size=0)
    expected_msg = r".*Array feat_type must only contain strings.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True]*100)

    cls = AutoSklearnClassifier(ensemble_size=0)
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
    cls.automl_ = unittest.mock.Mock()
    cls.automl_.InputValidator = unittest.mock.Mock()
    cls.automl_.InputValidator.target_validator = unittest.mock.Mock()

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

    params = cls.get_params()
    original_params = copy.deepcopy(params)

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

    # Compare the state of the model parameters with the original parameters
    new_params = clone(cls).get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # Taken from Sklearn code:
        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (cls, param_name, original_value, new_value))

    # Comply with https://scikit-learn.org/dev/glossary.html#term-classes
    is_classifier(cls)
    assert hasattr(cls, 'classes_')


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
    classifier.InputValidator = InputValidator(is_classification=True)
    classifier.InputValidator.target_validator.fit(
        pd.DataFrame(expected_result, dtype='category'),
    )
    classifier.InputValidator._is_fitted = True

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

    predict_mock.return_value = np.array(predicted_probabilities)

    classifier = AutoMLClassifier(
        time_left_for_this_task=1,
        per_run_time_limit=1,
        backend=backend,
        dask_client=dask_client,
    )
    classifier.InputValidator = InputValidator(is_classification=True)
    classifier.InputValidator.target_validator.fit(
        pd.DataFrame(predicted_indexes, dtype='int64'),
    )
    classifier.InputValidator._is_fitted = True

    assert classifier.InputValidator.target_validator.type_of_target == 'multilabel-indicator'

    actual_result = classifier.predict([None] * len(predicted_indexes))

    np.testing.assert_array_equal(predicted_indexes, actual_result)


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
    prediction = automl.predict(X)
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


def test_autosklearn_classification_methods_returns_self(dask_client):
    """
    Currently this method only tests that the methods of AutoSklearnClassifier
    is able to fit using fit(), fit_ensemble() and refit()
    """
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


@pytest.mark.parametrize("class_", [AutoSklearnClassifier, AutoSklearnRegressor,
                                    AutoSklearn2Classifier])
def test_check_estimator_signature(class_):
    # Make sure signature is store in self
    expected_subclass = ClassifierMixin if 'Classifier' in str(class_) else RegressorMixin
    assert issubclass(class_, expected_subclass)
    estimator = class_()
    for expected in list(inspect.signature(class_).parameters):
        assert hasattr(estimator, expected)


@pytest.mark.parametrize("selector_path", [None,  # No XDG_CACHE_HOME provided
                                           '/',  # XDG_CACHE_HOME has no permission
                                           tempfile.gettempdir(),  # in the user cache
                                           ])
def test_selector_file_askl2_can_be_created(selector_path):
    with unittest.mock.patch('os.environ.get') as mock_foo:
        mock_foo.return_value = selector_path
        if selector_path is not None and not os.access(selector_path, os.W_OK):
            with pytest.raises(PermissionError):
                importlib.reload(autosklearn.experimental.askl2)
        else:
            importlib.reload(autosklearn.experimental.askl2)
            assert os.path.exists(autosklearn.experimental.askl2.selector_file)
            if selector_path is None or not os.access(selector_path, os.W_OK):
                # We default to home in worst case
                assert os.path.expanduser("~") in str(autosklearn.experimental.askl2.selector_file)
            else:
                # a dir provided via XDG_CACHE_HOME
                assert selector_path in str(autosklearn.experimental.askl2.selector_file)
    # Re import it at the end so we do not affect other test
    importlib.reload(autosklearn.experimental.askl2)


def test_check_askl2_same_arguments_as_askl():
    # In case a new attribute is created, make sure it is there also in
    # ASKL2
    extra_arguments = list(set(
        inspect.getfullargspec(AutoSklearnEstimator.__init__).args) - set(
            inspect.getfullargspec(AutoSklearn2Classifier.__init__).args))
    expected_extra_args = ['exclude_estimators',
                           'include_preprocessors',
                           'resampling_strategy_arguments',
                           'exclude_preprocessors',
                           'include_estimators',
                           'get_smac_object_callback',
                           'initial_configurations_via_metalearning',
                           'resampling_strategy',
                           'metadata_directory']
    unexpected_args = set(extra_arguments) - set(expected_extra_args)
    assert len(unexpected_args) == 0, unexpected_args


@pytest.mark.parametrize("task_type", ['classification', 'regression'])
@pytest.mark.parametrize("resampling_strategy", ['test', 'cv', 'holdout'])
@pytest.mark.parametrize("disable_file_output", [True, False])
def test_fit_pipeline(dask_client, task_type, resampling_strategy, disable_file_output):
    """
    Tests that we can query the configuration space, and from the default configuration
    space, fit a classification pipeline with an acceptable score
    """
    X_train, y_train, X_test, y_test = putil.get_dataset(
        'iris' if task_type == 'classification' else 'boston'
    )
    estimator = AutoSklearnClassifier if task_type == 'classification' else AutoSklearnRegressor
    seed = 3
    automl = estimator(
        time_left_for_this_task=120,
        # Time left for task plays no role
        # only per run time limit
        per_run_time_limit=30,
        ensemble_size=0,
        dask_client=dask_client,
        include_estimators=['random_forest'],
        seed=seed,
        # We cannot get the configuration space with 'test' not fit with it
        resampling_strategy=resampling_strategy if resampling_strategy != 'test' else 'holdout',
    )
    config = automl.get_configuration_space(X_train, y_train,
                                            X_test=X_test, y_test=y_test,
                                            ).get_default_configuration()

    pipeline, run_info, run_value = automl.fit_pipeline(X=X_train, y=y_train, config=config,
                                                        X_test=X_test, y_test=y_test,
                                                        disable_file_output=disable_file_output,
                                                        resampling_strategy=resampling_strategy)

    assert isinstance(run_info.config, Configuration)
    assert run_info.cutoff == 30
    assert run_value.status == StatusType.SUCCESS, f"{run_info}->{run_value}"
    # We should produce a decent result
    assert run_value.cost < 0.2

    # Make sure that the pipeline can be pickled
    dump_file = os.path.join(tempfile.gettempdir(), 'automl.dump.pkl')
    with open(dump_file, 'wb') as f:
        pickle.dump(pipeline, f)

    if resampling_strategy == 'test' or disable_file_output:
        # We do not produce a pipeline in 'test'
        assert pipeline is None
    elif resampling_strategy == 'cv':
        # We should have fitted a Voting estimator
        assert hasattr(pipeline, 'estimators_')
    else:
        # We should have fitted a pipeline with named_steps
        assert hasattr(pipeline, 'named_steps')
        assert 'RandomForest' in pipeline.steps[-1][-1].choice.__class__.__name__

    # Num run should be 2, as 1 is for dummy classifier and we have not launch
    # another pipeline
    num_run = 2

    # Check the re-sampling strategy
    num_run_dir = automl.automl_._backend.get_numrun_directory(
        seed, num_run, budget=0.0)
    cv_model_path = os.path.join(num_run_dir, automl.automl_._backend.get_cv_model_filename(
        seed, num_run, budget=0.0))
    model_path = os.path.join(num_run_dir, automl.automl_._backend.get_model_filename(
        seed, num_run, budget=0.0))
    if resampling_strategy == 'test' or disable_file_output:
        # No file output is expected
        assert not os.path.exists(num_run_dir)
    else:
        # We expect the model path always
        # And the cv model only on 'cv'
        assert os.path.exists(model_path)
        if resampling_strategy == 'cv':
            assert os.path.exists(cv_model_path)
        elif resampling_strategy == 'holdout':
            assert not os.path.exists(cv_model_path)
