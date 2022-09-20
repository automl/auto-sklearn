from typing import Any, Dict, Sequence, Type, Union, cast

import copy
import glob
import importlib
import inspect
import itertools
import os
import pickle
import re
import tempfile

import joblib
import numpy as np
import numpy.ma as npma
import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.dummy
import sklearn.model_selection as model_selection
from ConfigSpace.configuration_space import Configuration
from dask.distributed import Client
from joblib import cpu_count
from sklearn.base import ClassifierMixin, RegressorMixin, clone, is_classifier
from smac.tae import StatusType

import autosklearn.estimators  # noqa F401
import autosklearn.pipeline.util as putil
from autosklearn.automl import AutoMLClassifier
from autosklearn.data.validation import InputValidator
from autosklearn.ensemble_building.run import Run
from autosklearn.ensembles import EnsembleSelection, MultiObjectiveDummyEnsemble
from autosklearn.estimators import (
    AutoSklearnClassifier,
    AutoSklearnEstimator,
    AutoSklearnRegressor,
)
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.metrics import accuracy, f1_macro, mean_squared_error, r2
from autosklearn.smbo import get_smac_object

import pytest
import unittest
import unittest.mock

import test.conftest
from test.test_automl.automl_utils import (
    count_succeses,
    include_single_scores,
    includes_all_scores,
    includes_train_scores,
    performance_over_time_is_plausible,
    print_debug_information,
)


def test_fit_n_jobs(tmp_dir):
    X_train, Y_train, X_test, Y_test = putil.get_dataset("breast_cancer")

    # test parallel Classifier to predict classes, not only indices
    Y_train += 1
    Y_test += 1

    n_jobs = 2

    class get_smac_object_wrapper:
        def __call__(self, *args, **kwargs):
            self.n_jobs = kwargs["n_jobs"]
            smac = get_smac_object(*args, **kwargs)
            self.dask_n_jobs = smac.solver.tae_runner.n_workers
            self.dask_client_n_jobs = len(
                smac.solver.tae_runner.client.scheduler_info()["workers"]
            )
            return smac

    get_smac_object_wrapper_instance = get_smac_object_wrapper()

    automl = AutoSklearnClassifier(
        delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=1,
        n_jobs=n_jobs,
        get_smac_object_callback=get_smac_object_wrapper_instance,
        max_models_on_disc=None,
    )
    automl.fit(X_train, Y_train)

    # Test that the argument is correctly passed to SMAC
    assert get_smac_object_wrapper_instance.n_jobs == n_jobs
    assert get_smac_object_wrapper_instance.dask_n_jobs == n_jobs
    assert get_smac_object_wrapper_instance.dask_client_n_jobs == n_jobs

    runhistory_data = automl.automl_.runhistory_.data

    successful_runs = {
        run_value.additional_info["num_run"]
        for run_value in runhistory_data.values()
        if (
            run_value.additional_info is not None
            and "num_run" in run_value.additional_info
            and run_value.status == StatusType.SUCCESS
        )
    }

    available_predictions = set()
    predictions = glob.glob(
        os.path.join(
            automl.automl_._backend.get_runs_directory(), "*", "predictions_ensemble_*"
        )
    )
    seeds = set()
    for prediction in predictions:
        prediction = os.path.split(prediction)[1]
        match = re.match(Run.RE_MODEL_PREDICTION_FILE, prediction)
        if match:
            print(match)
            seed, num_run, _ = match.groups()
            available_predictions.add(int(num_run))
            seeds.add(int(seed))

    # Remove the dummy prediction, it is not part of the runhistory
    if 1 in available_predictions:
        available_predictions.remove(1)

    # Make sure all predictions available are associated with a successful run
    # Don't want a rogue prediction file
    assert available_predictions <= successful_runs

    assert len(seeds) == 1

    ensemble_dir = automl.automl_._backend.get_ensemble_dir()
    ensembles = os.listdir(ensemble_dir)

    seeds = set()
    for ensemble_file in ensembles:
        seeds.add(int(ensemble_file.split(".")[0].split("_")[0]))
    assert len(seeds) == 1

    assert count_succeses(automl.cv_results_) > 0
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_feat_type_wrong_arguments():

    # Every Auto-Sklearn estimator has a backend, that allows a single
    # call to fit
    X = np.zeros((100, 100))
    y = np.zeros((100,))

    cls = AutoSklearnClassifier()
    expected_msg = r".*feat_type does not have same number of "
    "variables as X has features. 1 vs 100.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True])

    cls = AutoSklearnClassifier()
    expected_msg = r".*feat_type must only contain strings.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=[True] * 100)

    cls = AutoSklearnClassifier()
    expected_msg = r".*Only `Categorical`, `Numerical` and `String` are"
    "valid feature types, you passed `Car`.*"
    with pytest.raises(ValueError, match=expected_msg):
        cls.fit(X=X, y=y, feat_type=["Car"] * 100)


# Mock AutoSklearnEstimator.fit so the test doesn't actually run fit().
@unittest.mock.patch("autosklearn.estimators.AutoSklearnEstimator.fit")
def test_type_of_target(mock_estimator):
    # Test that classifier raises error for illegal target types.
    X = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ]
    )
    # Possible target types
    y_binary = np.array([0, 0, 1, 1])
    y_continuous = np.array([0.1, 1.3, 2.1, 4.0])
    y_multiclass = np.array([0, 1, 2, 0])
    y_multilabel = np.array(
        [
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ]
    )
    y_multiclass_multioutput = np.array(
        [
            [0, 1],
            [1, 3],
            [2, 2],
            [5, 3],
        ]
    )
    y_continuous_multioutput = np.array(
        [
            [0.1, 1.5],
            [1.2, 3.5],
            [2.7, 2.7],
            [5.5, 3.9],
        ]
    )

    cls = AutoSklearnClassifier()
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
        pytest.fail("cls.fit() raised ValueError while fitting " "binary targets")

    try:
        cls.fit(X, y_multiclass)
    except ValueError:
        pytest.fail("cls.fit() raised ValueError while fitting " "multiclass targets")

    try:
        cls.fit(X, y_multilabel)
    except ValueError:
        pytest.fail(
            "cls.fit() raised ValueError while fitting " "multilabel-indicator targets"
        )

    # Test that regressor raises error for illegal target types.
    reg = AutoSklearnRegressor()
    # Illegal target types for regression: multilabel-indicator
    # multiclass-multioutput
    expected_msg = r".*Regression with data of type"
    " multilabel-indicator is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        reg.fit(
            X=X,
            y=y_multilabel,
        )

    expected_msg = r".*Regression with data of type"
    " multiclass-multioutput is not supported.*"
    with pytest.raises(ValueError, match=expected_msg):
        reg.fit(
            X=X,
            y=y_multiclass_multioutput,
        )

    # Legal target types: continuous, multiclass,
    # continuous-multioutput,
    # binary
    try:
        reg.fit(X, y_continuous)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting " "continuous targets")

    try:
        reg.fit(X, y_multiclass)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting " "multiclass targets")

    try:
        reg.fit(X, y_continuous_multioutput)
    except ValueError:
        pytest.fail(
            "reg.fit() raised ValueError while fitting "
            "continuous_multioutput targets"
        )

    try:
        reg.fit(X, y_binary)
    except ValueError:
        pytest.fail("reg.fit() raised ValueError while fitting " "binary targets")


def test_performance_over_time_no_ensemble(tmp_dir):
    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris")

    cls = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=1,
        ensemble_class=None,
        initial_configurations_via_metalearning=0,
    )

    cls.fit(X_train, Y_train, X_test, Y_test)

    performance_over_time = cls.performance_over_time_
    assert include_single_scores(performance_over_time.columns) is True
    assert performance_over_time_is_plausible(performance_over_time) is True


def test_cv_results(tmp_dir):
    # TODO restructure and actually use real SMAC output from a long run
    # to do this unittest!
    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris")

    cls = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=1,
        initial_configurations_via_metalearning=0,
        scoring_functions=[autosklearn.metrics.precision, autosklearn.metrics.roc_auc],
    )

    params = cls.get_params()
    original_params = copy.deepcopy(params)

    cls.fit(X_train, Y_train)

    cv_results = cls.cv_results_
    assert isinstance(cv_results, dict), type(cv_results)
    assert isinstance(cv_results["mean_test_score"], np.ndarray), type(
        cv_results["mean_test_score"]
    )
    assert isinstance(cv_results["mean_fit_time"], np.ndarray), type(
        cv_results["mean_fit_time"]
    )
    assert isinstance(cv_results["params"], list), type(cv_results["params"])
    assert isinstance(cv_results["rank_test_scores"], np.ndarray), type(
        cv_results["rank_test_scores"]
    )
    assert isinstance(cv_results["metric_precision"], npma.MaskedArray), type(
        cv_results["metric_precision"]
    )
    assert isinstance(cv_results["metric_roc_auc"], npma.MaskedArray), type(
        cv_results["metric_roc_auc"]
    )
    cv_result_items = [
        isinstance(val, npma.MaskedArray)
        for key, val in cv_results.items()
        if key.startswith("param_")
    ]
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
            % (cls, param_name, original_value, new_value)
        )

    # Comply with https://scikit-learn.org/dev/glossary.html#term-classes
    is_classifier(cls)
    assert hasattr(cls, "classes_")


def test_cv_results_multi_objective(tmp_dir):
    # TODO restructure and actually use real SMAC output from a long run
    # to do this unittest!
    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris")

    cls = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=1,
        initial_configurations_via_metalearning=0,
        metric=[autosklearn.metrics.precision_macro, autosklearn.metrics.recall_macro],
        scoring_functions=[
            autosklearn.metrics.accuracy,
            autosklearn.metrics.balanced_accuracy,
        ],
    )

    params = cls.get_params()
    original_params = copy.deepcopy(params)

    cls.fit(X_train, Y_train)

    cv_results = cls.cv_results_
    assert isinstance(cv_results, dict), type(cv_results)
    assert "mean_test_score" not in cv_results
    assert "rank_test_scores" not in cv_results
    for expected_column in (
        "mean_test_precision_macro",
        "mean_test_recall_macro",
        "mean_fit_time",
        "rank_test_precision_macro",
        "rank_test_recall_macro",
        "metric_balanced_accuracy",
        "metric_accuracy",
    ):
        assert isinstance(cv_results[expected_column], np.ndarray), type(
            cv_results[expected_column]
        )

    assert isinstance(cv_results["params"], list), type(cv_results["params"])
    cv_result_items = [
        isinstance(val, npma.MaskedArray)
        for key, val in cv_results.items()
        if key.startswith("param_")
    ]
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
            % (cls, param_name, original_value, new_value)
        )

    # Comply with https://scikit-learn.org/dev/glossary.html#term-classes
    is_classifier(cls)
    assert hasattr(cls, "classes_")


@pytest.mark.parametrize(
    "estimator_type,dataset_name",
    [(AutoSklearnClassifier, "iris"), (AutoSklearnRegressor, "boston")],
)
def test_leaderboard(
    tmp_dir: str, estimator_type: Type[AutoSklearnEstimator], dataset_name: str
):
    # Comprehensive test tasks a substantial amount of time, manually set if
    # required.
    MAX_COMBO_SIZE_FOR_INCLUDE_PARAM = 3  # [0, len(valid_columns) + 1]
    column_types = AutoSklearnEstimator._leaderboard_columns(num_metrics=1)

    # Create a dict of all possible param values for each param
    # with some invalid one's of the incorrect type
    include_combinations = itertools.chain(
        itertools.combinations(column_types["all"], item_count)
        for item_count in range(1, MAX_COMBO_SIZE_FOR_INCLUDE_PARAM)
    )
    valid_params = {
        "detailed": [True, False],
        "ensemble_only": [True, False],
        "top_k": [-10, 0, 1, 10, "all"],
        "sort_by": [*column_types["all"], "invalid"],
        "sort_order": ["ascending", "descending", "auto", "invalid", None],
        "include": itertools.chain([None, "invalid", "type"], include_combinations),
    }

    # Create a generator of all possible combinations of valid_params
    params_generator = iter(
        dict(zip(valid_params.keys(), param_values))
        for param_values in itertools.product(*valid_params.values())
    )

    X_train, Y_train, _, _ = putil.get_dataset(dataset_name)
    model = estimator_type(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=1,
    )

    model.fit(X_train, Y_train)

    for params in params_generator:
        # Convert from iterator to solid list
        if params["include"] is not None and not isinstance(params["include"], str):
            params["include"] = list(params["include"])

        # Invalid top_k should raise an error, is a positive int or 'all'
        if not (params["top_k"] == "all" or params["top_k"] > 0):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Invalid sort_by column
        elif params["sort_by"] not in column_types["all"]:
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Shouldn't accept an invalid sort order
        elif params["sort_order"] not in ["ascending", "descending", "auto"]:
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # include is single str but not valid
        elif (
            isinstance(params["include"], str)
            and params["include"] not in column_types["all"]
        ):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Crash if include is list but contains invalid column
        elif (
            isinstance(params["include"], list)
            and len(set(params["include"]) - set(column_types["all"])) != 0
        ):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Can't have just model_id, in both single str and list case
        elif params["include"] == "model_id" or params["include"] == ["model_id"]:
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Else all valid combinations should be validated
        else:
            leaderboard = model.leaderboard(**params)

            # top_k should never be less than the rows given back
            # It can however be larger
            if isinstance(params["top_k"], int):
                assert params["top_k"] >= len(leaderboard)

            # Check the right columns are present and in the right order
            # The model_id is set as the index, not included in pandas columns
            columns = list(leaderboard.columns)

            def exclude(lst, s):
                return [x for x in lst if x != s]

            if params["include"] is not None:
                # Include with only single str should be the only column
                if isinstance(params["include"], str):
                    assert params["include"] in columns and len(columns) == 1
                # Include as a list should have all the columns without model_id
                else:
                    assert columns == exclude(params["include"], "model_id")
            elif params["detailed"]:
                assert columns == exclude(column_types["detailed"], "model_id")
            else:
                assert columns == exclude(column_types["simple"], "model_id")

            # Ensure that if it's ensemble only
            # Can only check if 'ensemble_weight' is present
            if params["ensemble_only"] and "ensemble_weight" in columns:
                assert all(leaderboard["ensemble_weight"] > 0)


@pytest.mark.parametrize(
    "estimator_type,dataset_name,metrics",
    [
        (
            AutoSklearnClassifier,
            "iris",
            (autosklearn.metrics.accuracy, autosklearn.metrics.balanced_accuracy),
        ),
        (
            AutoSklearnRegressor,
            "boston",
            (autosklearn.metrics.r2, autosklearn.metrics.root_mean_squared_error),
        ),
    ],
)
def test_leaderboard_multi_objective(
    tmp_dir: str,
    estimator_type: Type[AutoSklearnEstimator],
    dataset_name: str,
    metrics: Sequence[autosklearn.metrics.Scorer],
):
    # Comprehensive test tasks a substantial amount of time, manually set if
    # required.
    MAX_COMBO_SIZE_FOR_INCLUDE_PARAM = 3  # [0, len(valid_columns) + 1]
    column_types = AutoSklearnEstimator._leaderboard_columns(num_metrics=2)

    # Create a dict of all possible param values for each param
    # with some invalid one's of the incorrect type
    include_combinations = itertools.chain(
        itertools.combinations(column_types["all"], item_count)
        for item_count in range(1, MAX_COMBO_SIZE_FOR_INCLUDE_PARAM)
    )
    valid_params = {
        "detailed": [True, False],
        "ensemble_only": [True, False],
        "top_k": [-10, 0, 1, 10, "all"],
        "sort_by": [
            "cost",
            "cost_0",
            "cost_1",
            ["cost_1", "cost_0"],
            *column_types["all"],
            "invalid",
        ],
        "sort_order": ["ascending", "descending", "auto", "invalid", None],
        "include": itertools.chain([None, "invalid", "type"], include_combinations),
    }

    # Create a generator of all possible combinations of valid_params
    params_generator = iter(
        dict(zip(valid_params.keys(), param_values))
        for param_values in itertools.product(*valid_params.values())
    )

    X_train, Y_train, _, _ = putil.get_dataset(dataset_name)
    model = estimator_type(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        seed=test.conftest.DEFAULT_SEED,
        metric=metrics,
    )

    model.fit(X_train, Y_train)

    for params in params_generator:
        # Convert from iterator to solid list
        if params["include"] is not None and not isinstance(params["include"], str):
            params["include"] = list(params["include"])

        # Invalid top_k should raise an error, is a positive int or 'all'
        if not (params["top_k"] == "all" or params["top_k"] > 0):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Invalid sort_by column
        elif (
            params["sort_by"] not in column_types["all"]
            and params["sort_by"] != "cost"
            and params["sort_by"] != ["cost_1", "cost_0"]
            and params["sort_by"] not in ["cost_0", "cost_1"]
        ):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Shouldn't accept an invalid sort order
        elif params["sort_order"] not in ["ascending", "descending", "auto"]:
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # include is single str but not valid
        elif (
            isinstance(params["include"], str)
            and params["include"] not in column_types["all"]
        ):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Crash if include is list but contains invalid column
        elif (
            isinstance(params["include"], list)
            and len(set(params["include"]) - set(column_types["all"])) != 0
        ):
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Can't have just model_id, in both single str and list case
        elif params["include"] == "model_id" or params["include"] == ["model_id"]:
            with pytest.raises(ValueError):
                model.leaderboard(**params)

        # Else all valid combinations should be validated
        else:
            leaderboard = model.leaderboard(**params)
            assert "cost" not in leaderboard.columns

            if params["include"] is None:
                assert "cost_0" in leaderboard.columns
                assert "cost_1" in leaderboard.columns
            else:
                for cost_name in ["cost_0", "cost_1"]:
                    if cost_name in params["include"]:
                        assert cost_name in leaderboard.columns

            # top_k should never be less than the rows given back
            # It can however be larger
            if isinstance(params["top_k"], int):
                assert params["top_k"] >= len(leaderboard)

            # Check the right columns are present and in the right order
            # The model_id is set as the index, not included in pandas columns
            columns = list(leaderboard.columns)

            def exclude(lst, s):
                return [x for x in lst if x != s]

            if params["include"] is not None:
                # Include with only single str should be the only column
                if isinstance(params["include"], str):
                    assert params["include"] in columns and len(columns) == 1
                # Include as a list should have all the columns without model_id
                else:
                    assert columns == exclude(params["include"], "model_id")
            elif params["detailed"]:
                assert columns == exclude(column_types["detailed"], "model_id")
            else:
                assert columns == exclude(column_types["simple"], "model_id")

            # Ensure that if it's ensemble only
            # Can only check if 'ensemble_weight' is present
            if params["ensemble_only"] and "ensemble_weight" in columns:
                assert all(leaderboard["ensemble_weight"] > 0)


@pytest.mark.parametrize("estimator", [AutoSklearnRegressor])
@pytest.mark.parametrize("resampling_strategy", ["holdout"])
@pytest.mark.parametrize(
    "X",
    [
        np.asarray(
            [[1.0, 1.0, 1.0]] * 25
            + [[2.0, 2.0, 2.0]] * 25
            + [[3.0, 3.0, 3.0]] * 25
            + [[4.0, 4.0, 4.0]] * 25
        )
    ],
)
@pytest.mark.parametrize(
    "y", [np.asarray([1.0] * 25 + [2.0] * 25 + [3.0] * 25 + [4.0] * 25)]
)
def test_show_models_with_holdout(
    tmp_dir: str,
    dask_client: Client,
    estimator: AutoSklearnEstimator,
    resampling_strategy: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Parameters
    ----------
    tmp_dir: str
        The temporary directory to use for this test

    dask_client: dask.distributed.Client
         The dask client to use for this test

    estimator: AutoSklearnEstimator
         The estimator to train

    resampling_strategy: str
         The resampling strategy to use

    X: np.ndarray
        The X data to use for this estimator

    y: np.ndarray
         The targets to use for this estimator

    Expects
    -------
    * Expects all the model dictionaries to have ``model_keys``
    * Expects all models to have an auto-sklearn wrapped model ``regressor``
    * Expects all models to have a sklearn wrapped model ``sklearn_regressor``
    * Expects no model to have any ``None`` value
    """

    automl = estimator(
        time_left_for_this_task=60,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        resampling_strategy=resampling_strategy,
        dask_client=dask_client,
    )
    automl.fit(X, y)

    models = automl.show_models().values()

    model_keys = set(
        [
            "model_id",
            "rank",
            "cost",
            "ensemble_weight",
            "data_preprocessor",
            "feature_preprocessor",
            "regressor",
            "sklearn_regressor",
        ]
    )

    assert all([model_keys == set(model.keys()) for model in models])
    assert all([model["regressor"] for model in models])
    assert all([model["sklearn_regressor"] for model in models])
    assert not any([None in model.values() for model in models])


@pytest.mark.parametrize("estimator", [AutoSklearnClassifier])
@pytest.mark.parametrize("resampling_strategy", ["cv"])
@pytest.mark.parametrize(
    "X", [np.asarray([[1.0, 1.0, 1.0]] * 50 + [[2.0, 2.0, 2.0]] * 50)]
)
@pytest.mark.parametrize("y", [np.asarray([1] * 50 + [2] * 50)])
def test_show_models_with_cv(
    tmp_dir: str,
    dask_client: Client,
    estimator: AutoSklearnEstimator,
    resampling_strategy: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Parameters
    ----------
    tmp_dir: str
        The temporary directory to use for this test

    dask_client: dask.distributed.Client
         The dask client to use for this test

    estimator: AutoSklearnEstimator
         The estimator to train

    resampling_strategy: str
         The resampling strategy to use

    X: np.ndarray
        The X data to use for this estimator

    y: np.ndarray
         The targets to use for this estimator

    Expects
    -------
    * Expects all the model dictionaries to have `model_keys`
    * Expects no model to have any `None` value
    * Expects all the estimators in a model to have `estimator_keys`
    * Expects all model estimators to have an auto-sklearn wrapped model `classifier`
    * Expects all model estimators to have a sklearn wrapped model `sklearn_classifier`
    * Expects no estimator to have None
    """

    automl = estimator(
        time_left_for_this_task=120,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        resampling_strategy=resampling_strategy,
        dask_client=dask_client,
    )
    automl.fit(X, y)

    models = automl.show_models().values()

    model_keys = set(
        ["model_id", "rank", "cost", "ensemble_weight", "voting_model", "estimators"]
    )

    estimator_keys = set(
        [
            "data_preprocessor",
            "balancing",
            "feature_preprocessor",
            "classifier",
            "sklearn_classifier",
        ]
    )

    assert all([model_keys == set(model.keys()) for model in models])
    assert not any([None in model.values() for model in models])
    assert all(
        [
            estimator_keys == set(estimator.keys())
            for model in models
            for estimator in model["estimators"]
        ]
    )
    assert all(
        [
            estimator["classifier"]
            for model in models
            for estimator in model["estimators"]
        ]
    )
    assert all(
        [
            estimator["sklearn_classifier"]
            for model in models
            for estimator in model["estimators"]
        ]
    )
    assert not any(
        [
            None in estimator.values()
            for model in models
            for estimator in model["estimators"]
        ]
    )


@unittest.mock.patch("autosklearn.estimators.AutoSklearnEstimator.build_automl")
def test_fit_n_jobs_negative(build_automl_patch):
    n_cores = cpu_count()
    cls = AutoSklearnEstimator(n_jobs=-1)
    cls.fit()
    assert cls._n_jobs == n_cores


def test_get_number_of_available_cores():
    n_cores = cpu_count()
    assert n_cores >= 1, n_cores


@unittest.mock.patch("autosklearn.automl.AutoML.predict")
def test_multiclass_prediction(predict_mock, dask_client):
    predicted_probabilities = [
        [0, 0, 0.99],
        [0, 0.99, 0],
        [0.99, 0, 0],
        [0, 0.99, 0],
        [0, 0, 0.99],
    ]
    predicted_indexes = [2, 1, 0, 1, 2]
    expected_result = ["c", "b", "a", "b", "c"]

    predict_mock.return_value = np.array(predicted_probabilities)

    classifier = AutoMLClassifier(
        time_left_for_this_task=1,
        per_run_time_limit=1,
        dask_client=dask_client,
    )
    classifier.InputValidator = InputValidator(is_classification=True)
    classifier.InputValidator.target_validator.fit(
        pd.DataFrame(expected_result, dtype="category"),
    )
    classifier.InputValidator._is_fitted = True

    actual_result = classifier.predict([None] * len(predicted_indexes))

    np.testing.assert_array_equal(expected_result, actual_result)


@unittest.mock.patch("autosklearn.automl.AutoML.predict")
def test_multilabel_prediction(predict_mock, dask_client):
    predicted_probabilities = [
        [0.99, 0],
        [0.99, 0],
        [0, 0.99],
        [0.99, 0.99],
        [0.99, 0.99],
    ]
    predicted_indexes = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [1, 1]])

    predict_mock.return_value = np.array(predicted_probabilities)

    classifier = AutoMLClassifier(
        time_left_for_this_task=1,
        per_run_time_limit=1,
        dask_client=dask_client,
    )
    classifier.InputValidator = InputValidator(is_classification=True)
    classifier.InputValidator.target_validator.fit(
        pd.DataFrame(predicted_indexes, dtype="int64"),
    )
    classifier.InputValidator._is_fitted = True

    assert (
        classifier.InputValidator.target_validator.type_of_target
        == "multilabel-indicator"
    )

    actual_result = classifier.predict([None] * len(predicted_indexes))

    np.testing.assert_array_equal(predicted_indexes, actual_result)


def test_can_pickle_classifier(tmp_dir, dask_client):
    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris")
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        delete_tmp_folder_after_terminate=False,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        dask_client=dask_client,
    )

    automl.fit(X_train, Y_train)

    initial_predictions = automl.predict(X_test)
    initial_accuracy = sklearn.metrics.accuracy_score(Y_test, initial_predictions)
    assert initial_accuracy >= 0.75
    assert count_succeses(automl.cv_results_) > 0
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True

    # Test pickle
    dump_file = os.path.join(tmp_dir, "automl.dump.pkl")

    with open(dump_file, "wb") as f:
        pickle.dump(automl, f)

    with open(dump_file, "rb") as f:
        restored_automl = pickle.load(f)

    restored_predictions = restored_automl.predict(X_test)
    restored_accuracy = sklearn.metrics.accuracy_score(Y_test, restored_predictions)
    assert restored_accuracy >= 0.75
    assert initial_accuracy == restored_accuracy

    # Test joblib
    dump_file = os.path.join(tmp_dir, "automl.dump.joblib")

    joblib.dump(automl, dump_file)

    restored_automl = joblib.load(dump_file)

    restored_predictions = restored_automl.predict(X_test)
    restored_accuracy = sklearn.metrics.accuracy_score(Y_test, restored_predictions)
    assert restored_accuracy >= 0.75
    assert initial_accuracy == restored_accuracy


def test_multilabel(tmp_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris", make_multilabel=True)
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        dask_client=dask_client,
    )

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (50, 3), print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True

    score = f1_macro(Y_test, predictions)
    assert score >= 0.9, print_debug_information(automl)

    probs = automl.predict_proba(X_train)
    assert np.mean(probs) == pytest.approx(0.33, rel=1e-1)


def test_binary(tmp_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset("iris", make_binary=True)
    automl = AutoSklearnClassifier(
        time_left_for_this_task=40,
        delete_tmp_folder_after_terminate=False,
        per_run_time_limit=10,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        dask_client=dask_client,
    )

    automl.fit(
        X_train,
        Y_train,
        X_test=X_test,
        y_test=Y_test,
        dataset_name="binary_test_dataset",
    )

    predictions = automl.predict(X_test)
    assert predictions.shape == (50,), print_debug_information(automl)

    score = accuracy(Y_test, predictions)
    assert score > 0.9, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)
    assert includes_all_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_classification_pandas_support(tmp_dir, dask_client):

    X, y = sklearn.datasets.fetch_openml(
        data_id=2,  # cat/num dataset
        return_X_y=True,
        as_frame=True,
    )

    # Drop NAN!!
    X = X.dropna(axis="columns")

    # This test only make sense if input is dataframe
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        exclude={"classifier": ["libsvm_svc"]},
        dask_client=dask_client,
        seed=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
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
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_regression(tmp_dir, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset("boston")
    automl = AutoSklearnRegressor(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=os.path.join(tmp_dir, "backend"),
        dask_client=dask_client,
    )

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (356,)
    score = mean_squared_error(Y_test, predictions)

    # On average np.sqrt(30) away from the target -> ~5.5 on average
    # Results with select rates drops avg score to a range of -32.40 to -37,
    # on 30 seconds constraint.
    # With more time_left_for_this_task this is no longer an issue
    assert score >= -37, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_cv_regression(tmp_dir, dask_client):
    """
    Makes sure that when using a cv strategy, we are able to fit
    a regressor
    """

    X_train, Y_train, X_test, Y_test = putil.get_dataset(
        "boston", train_size_maximum=300
    )
    automl = AutoSklearnRegressor(
        time_left_for_this_task=60,
        per_run_time_limit=10,
        resampling_strategy="cv",
        tmp_folder=os.path.join(tmp_dir, "backend"),
        dask_client=dask_client,
    )

    automl.fit(X_train, Y_train)

    predictions = automl.predict(X_test)
    assert predictions.shape == (206,)
    score = r2(Y_test, predictions)
    assert score >= 0.1, print_debug_information(automl)
    assert count_succeses(automl.cv_results_) > 0, print_debug_information(automl)
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_regression_pandas_support(tmp_dir, dask_client):

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
        tmp_folder=os.path.join(tmp_dir, "backend"),
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
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True


def test_autosklearn_classification_methods_returns_self(dask_client):
    """Currently this method only tests that the methods of AutoSklearnClassifier
    is able to fit using fit(), fit_ensemble() and refit()
    """
    X_train, y_train, X_test, y_test = putil.get_dataset("iris")
    automl = AutoSklearnClassifier(
        time_left_for_this_task=60,
        delete_tmp_folder_after_terminate=False,
        per_run_time_limit=10,
        dask_client=dask_client,
        exclude={"feature_preprocessor": ["fast_ica"]},
    )

    automl_fitted = automl.fit(X_train, y_train)

    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(
        y_train, ensemble_kwargs={"ensemble_size": 5}
    )
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())

    assert automl is automl_refitted


# Currently this class only tests that the methods of AutoSklearnRegressor
# that should return self actually return self.
def test_autosklearn_regression_methods_returns_self(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset("boston")
    automl = AutoSklearnRegressor(
        time_left_for_this_task=30,
        delete_tmp_folder_after_terminate=False,
        per_run_time_limit=5,
        dask_client=dask_client,
    )

    automl_fitted = automl.fit(X_train, y_train)
    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(
        y_train, ensemble_kwargs={"ensemble_size": 5}
    )
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())
    assert automl is automl_refitted


def test_autosklearn2_classification_methods_returns_self(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset("iris")
    automl = AutoSklearn2Classifier(
        time_left_for_this_task=60,
        delete_tmp_folder_after_terminate=False,
        dask_client=dask_client,
    )

    automl_fitted = automl.fit(X_train, y_train)

    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(
        y_train, ensemble_kwargs={"ensemble_size": 5}
    )
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())

    assert automl is automl_refitted

    predictions = automl_fitted.predict(X_test)
    assert (
        sklearn.metrics.accuracy_score(y_test, predictions) >= 4 / 5
    ), print_debug_information(automl)

    pickle.dumps(automl_fitted)


def test_autosklearn2_classification_methods_returns_self_sparse(dask_client):
    X_train, y_train, X_test, y_test = putil.get_dataset(
        "breast_cancer", make_sparse=True
    )
    automl = AutoSklearn2Classifier(
        time_left_for_this_task=60,
        delete_tmp_folder_after_terminate=False,
        dask_client=dask_client,
    )

    automl_fitted = automl.fit(X_train, y_train)

    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(
        y_train, ensemble_kwargs={"ensemble_size": 5}
    )
    assert automl is automl_ensemble_fitted

    automl_refitted = automl.refit(X_train.copy(), y_train.copy())

    assert automl is automl_refitted

    predictions = automl_fitted.predict(X_test)
    assert (
        sklearn.metrics.accuracy_score(y_test, predictions) >= 4 / 5
    ), print_debug_information(automl)

    assert "boosting" not in str(automl.get_configuration_space(X=X_train, y=y_train))

    pickle.dumps(automl_fitted)


@pytest.mark.parametrize(
    "class_", [AutoSklearnClassifier, AutoSklearnRegressor, AutoSklearn2Classifier]
)
def test_check_estimator_signature(class_):
    # Make sure signature is store in self
    expected_subclass = (
        ClassifierMixin if "Classifier" in str(class_) else RegressorMixin
    )
    assert issubclass(class_, expected_subclass)
    estimator = class_()
    for expected in list(inspect.signature(class_).parameters):
        assert hasattr(estimator, expected)


@pytest.mark.parametrize(
    "selector_path",
    [
        None,  # No XDG_CACHE_HOME provided
        "/",  # XDG_CACHE_HOME has no permission
        tempfile.gettempdir(),  # in the user cache
    ],
)
def test_selector_file_askl2_can_be_created(selector_path):

    with unittest.mock.patch("os.environ.get") as mock_foo:
        mock_foo.return_value = selector_path
        if selector_path is not None and not os.access(selector_path, os.W_OK):
            with pytest.raises(PermissionError):
                importlib.reload(autosklearn.experimental.askl2)
                automl = AutoSklearn2Classifier(
                    time_left_for_this_task=60, delete_tmp_folder_after_terminate=False
                )

        else:
            importlib.reload(autosklearn.experimental.askl2)
            automl = AutoSklearn2Classifier(
                time_left_for_this_task=60, delete_tmp_folder_after_terminate=False
            )
            for metric in automl.selector_metrics:
                assert os.path.exists(automl.selector_files[metric.name])
                if selector_path is None or not os.access(selector_path, os.W_OK):
                    # We default to home in worst case
                    assert os.path.expanduser("~") in str(
                        automl.selector_files[metric.name]
                    )
                else:
                    # a dir provided via XDG_CACHE_HOME
                    assert selector_path in str(automl.selector_files[metric.name])
    # Re import it at the end so we do not affect other test
    importlib.reload(autosklearn.experimental.askl2)


@pytest.mark.parametrize(
    "metric",
    [metric for metric in autosklearn.experimental.askl2.selector_metrics],
)
def test_askl2_fits_selector_for_given_metrics_at_init(tmp_path, metric):

    assert tmp_path.is_dir()
    assert len(list(tmp_path.iterdir())) == 0
    temp_dir = str(tmp_path)

    with unittest.mock.patch("os.environ.get") as mock_foo:
        mock_foo.return_value = temp_dir
        automl = AutoSklearn2Classifier(
            time_left_for_this_task=60,
            delete_tmp_folder_after_terminate=False,
            metric=metric,
        )
        assert (
            len(automl.selector_files) == 1
        )  # only one selector file should have been created
        assert os.path.exists(
            str(automl.selector_files[metric.name])
        )  # check if the path exists

        # check if selector is retrained when
        # another object with the same metric is created
        automl_1 = AutoSklearn2Classifier(
            time_left_for_this_task=60,
            delete_tmp_folder_after_terminate=False,
            metric=metric,
        )
        assert (
            len(automl_1.selector_files) == 1
        )  # only one selector file should have been created
        assert os.path.exists(
            str(automl_1.selector_files[metric.name])
        )  # check if the path exists
        assert not automl_1.required_training


def test_askl2_fit_when_no_metric_specified(tmp_path):

    assert tmp_path.is_dir()
    assert len(list(tmp_path.iterdir())) == 0
    temp_dir = str(tmp_path)

    with unittest.mock.patch("os.environ.get") as mock_foo:
        mock_foo.return_value = temp_dir
        automl = AutoSklearn2Classifier(
            time_left_for_this_task=60, delete_tmp_folder_after_terminate=False
        )
        assert len(automl.selector_files) == 3
        for metric in automl.selector_metrics:
            assert os.path.exists(str(automl.selector_files[metric.name]))


def test_check_askl2_same_arguments_as_askl() -> None:
    """Check the asklearn2 has the same args as asklearn1

    Expects
    -------
    * The set of arguments for AutoSklearnClassifier is the same as
        AutoSklearn2Classifier except for a few expected arugments.
    """
    autosklearn1_classifier_args = set(
        inspect.getfullargspec(AutoSklearnEstimator.__init__).args
    )
    autosklearn2_classifier_args = set(
        inspect.getfullargspec(AutoSklearn2Classifier.__init__).args
    )
    extra_arguments = autosklearn1_classifier_args - autosklearn2_classifier_args

    expected_extra_args = set(
        [
            "exclude",
            "include",
            "resampling_strategy_arguments",
            "get_smac_object_callback",
            "initial_configurations_via_metalearning",
            "resampling_strategy",
            "metadata_directory",
            "get_trials_callback",
        ]
    )
    unexpected_args = extra_arguments - expected_extra_args

    assert len(unexpected_args) == 0, unexpected_args


@pytest.mark.parametrize("task_type", ["classification", "regression"])
@pytest.mark.parametrize("resampling_strategy", ["test", "cv", "holdout"])
@pytest.mark.parametrize("disable_file_output", [True, False])
def test_fit_pipeline(dask_client, task_type, resampling_strategy, disable_file_output):
    """
    Tests that we can query the configuration space, and from the default configuration
    space, fit a classification pipeline with an acceptable score
    """
    X_train, y_train, X_test, y_test = putil.get_dataset(
        "iris" if task_type == "classification" else "boston"
    )
    estimator = (
        AutoSklearnClassifier if task_type == "classification" else AutoSklearnRegressor
    )
    seed = 3
    if task_type == "classification":
        include = {"classifier": ["random_forest"]}
    else:
        include = {"regressor": ["random_forest"]}
    automl = estimator(
        delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=120,
        # Time left for task plays no role
        # only per run time limit
        per_run_time_limit=30,
        dask_client=dask_client,
        include=include,
        seed=seed,
        # We cannot get the configuration space with 'test' not fit with it
        resampling_strategy=resampling_strategy
        if resampling_strategy != "test"
        else "holdout",
    )
    config = automl.get_configuration_space(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
    ).get_default_configuration()

    pipeline, run_info, run_value = automl.fit_pipeline(
        X=X_train,
        y=y_train,
        config=config,
        X_test=X_test,
        y_test=y_test,
        disable_file_output=disable_file_output,
        resampling_strategy=resampling_strategy,
    )

    assert isinstance(run_info.config, Configuration)
    assert run_info.cutoff == 30
    assert run_value.status == StatusType.SUCCESS, f"{run_info}->{run_value}"
    # We should produce a decent result
    assert run_value.cost < 0.2

    # Make sure that the pipeline can be pickled
    dump_file = os.path.join(tempfile.gettempdir(), "automl.dump.pkl")
    with open(dump_file, "wb") as f:
        pickle.dump(pipeline, f)

    if resampling_strategy == "test" or disable_file_output:
        # We do not produce a pipeline in 'test'
        assert pipeline is None
    elif resampling_strategy == "cv":
        # We should have fitted a Voting estimator
        assert hasattr(pipeline, "estimators_")
    else:
        # We should have fitted a pipeline with named_steps
        assert hasattr(pipeline, "named_steps")
        assert "RandomForest" in pipeline.steps[-1][-1].choice.__class__.__name__

    # Num run should be 2, as 1 is for dummy classifier and we have not launch
    # another pipeline
    num_run = 2

    # Check the re-sampling strategy
    num_run_dir = automl.automl_._backend.get_numrun_directory(
        seed, num_run, budget=0.0
    )
    cv_model_path = os.path.join(
        num_run_dir,
        automl.automl_._backend.get_cv_model_filename(seed, num_run, budget=0.0),
    )
    model_path = os.path.join(
        num_run_dir,
        automl.automl_._backend.get_model_filename(seed, num_run, budget=0.0),
    )
    if resampling_strategy == "test" or disable_file_output:
        # No file output is expected
        assert not os.path.exists(num_run_dir)
    else:
        # We expect the model path always
        # And the cv model only on 'cv'
        assert os.path.exists(model_path)
        if resampling_strategy == "cv":
            assert os.path.exists(cv_model_path)
        elif resampling_strategy == "holdout":
            assert not os.path.exists(cv_model_path)


@pytest.mark.parametrize("data_type", ["pandas", "numpy"])
@pytest.mark.parametrize("include_categorical", [True, False])
def test_pass_categorical_and_numeric_columns_to_pipeline(
    dask_client, data_type, include_categorical
):
    # Prepare the training data
    X, y = sklearn.datasets.make_classification(random_state=0)
    X = cast(np.ndarray, X)

    n_features = X.shape[1]

    # If categorical, insert a row of 'categorical' '0's at last col
    if include_categorical:
        X = np.insert(X, n_features, values=0, axis=1)

    if data_type == "pandas":
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, dtype="category")

        # Set the last column to categorical
        if include_categorical:
            X.loc[:, n_features] = X.loc[:, n_features].astype("category")

    # Specify the feature_types
    if data_type == "numpy" and include_categorical:
        feat_type = ["numerical"] * n_features + ["categorical"]
    else:
        feat_type = None

    # Create the splits
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.5, random_state=3
    )

    # Create Estimator
    # Time left for task plays no role for fit_pipeline
    automl = AutoSklearnClassifier(
        delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=120,
        per_run_time_limit=30,
        seed=0,
        dask_client=dask_client,
        include={"classifier": ["random_forest"]},
    )

    config_space = automl.get_configuration_space(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        feat_type=feat_type,
    )
    config = config_space.get_default_configuration()

    pipeline, _, run_value = automl.fit_pipeline(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
        feat_type=feat_type,
    )

    assert pipeline is not None, "Expected a pipeline from automl.fit_pipeline"

    feature_validator = automl.automl_.InputValidator.feature_validator  # type: ignore
    transformed_X_test = feature_validator.transform(X_test)
    predictions = pipeline.predict(transformed_X_test)

    # We should produce a half decent result
    assert run_value.cost < 0.40, f"Run value:\n {run_value}"

    # Outputs should be the correct length
    assert np.shape(predictions)[0] == np.shape(y_test)[0]

    n_columns = np.shape(X)[1]

    if include_categorical:
        expected_feat_types = {
            i: feature_type
            for i, feature_type in enumerate(
                ["numerical"] * (n_columns - 1) + ["categorical"]
            )
        }

    else:
        expected_feat_types = {
            i: feature_type for i, feature_type in enumerate(["numerical"] * n_columns)
        }

    pipeline_feat_types = pipeline.named_steps["data_preprocessor"].choice.feat_type
    assert expected_feat_types == pipeline_feat_types


@pytest.mark.parametrize("as_frame", [True, False])
def test_autosklearn_anneal(as_frame):
    """
    This test makes sure that anneal dataset can be fitted and scored.
    This dataset is quite complex, with NaN, categorical and numerical columns
    so is a good testcase for unit-testing
    """
    X, y = sklearn.datasets.fetch_openml(data_id=2, return_X_y=True, as_frame=as_frame)
    automl = AutoSklearnClassifier(
        time_left_for_this_task=60,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        smac_scenario_args={"runcount_limit": 6},
        resampling_strategy="holdout-iterative-fit",
    )

    if as_frame:
        # Let autosklearn calculate the feat types
        automl_fitted = automl.fit(X, y)

    else:
        X_, y_ = sklearn.datasets.fetch_openml(
            data_id=2, return_X_y=True, as_frame=True
        )
        feat_type = [
            "categorical" if X_[col].dtype.name == "category" else "numerical"
            for col in X_.columns
        ]

        automl_fitted = automl.fit(X, y, feat_type=feat_type)

    assert automl is automl_fitted

    automl_ensemble_fitted = automl.fit_ensemble(
        y, ensemble_kwargs={"ensemble_size": 5}
    )
    assert automl is automl_ensemble_fitted

    # We want to make sure we can learn from this data.
    # This is a test to make sure the data format (numpy/pandas)
    # can be used in a meaningful way -- not meant for generalization,
    # hence we use the train dataset
    assert automl_fitted.score(X, y) > 0.75


@pytest.mark.parametrize(
    "dataset_compression", [False, True, {"memory_allocation": 0.2}]
)
def test_param_dataset_compression(dataset_compression: Union[bool, Dict[str, Any]]):
    """We expect this does not get parsed and modified until it gets to the AutoML
    class, In the meantime, it's value remains whatever was passed in.

    Expects
    -------
    * The private attribute should be set to what was passed
    """
    model = AutoSklearnClassifier(dataset_compression=dataset_compression)

    assert model.dataset_compression == dataset_compression


def test_ensemble_default_resolves():
    model = AutoSklearnClassifier(
        ensemble_class="default",
        metric=accuracy,
    )
    assert model.ensemble_class == EnsembleSelection
    model = AutoSklearnClassifier(
        ensemble_class="default",
        metric=[accuracy, f1_macro],
    )
    assert model.ensemble_class == MultiObjectiveDummyEnsemble
