"""Test the performance over time functionality of automl instances"""
from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases
from pytest_cases.filters import has_tag

import test.test_automl.cases as cases


@parametrize_with_cases(
    "automl",
    cases=cases,
    filter=has_tag("fitted") & ~has_tag("no_ensemble"),
)
def test_performance_over_time_with_ensemble(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl: AutoMLClassifier
        The fitted automl instance with an ensemble

    Expects
    -------
    * Performance over time should include only the given columns
    * The performance over time should have at least one entry that isn't NaN
    * The timestamps should be monotonic
    """
    expected_performance_columns = {
        "single_best_train_score",
        "single_best_optimization_score",
        "ensemble_optimization_score",
        "Timestamp",
    }
    columns = automl.performance_over_time_.columns
    assert set(columns) == set(expected_performance_columns)

    perf_over_time = automl.performance_over_time_
    assert len(perf_over_time.drop(columns="Timestamp").dropna()) != 0
    assert perf_over_time["Timestamp"].is_monotonic
