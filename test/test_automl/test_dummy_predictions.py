"""Test the dummy predictor of AutoML

Dummy models can serve as an early warning of issues with parameters during fit
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

from pathlib import Path

import numpy as np
from smac.tae import StatusType

from autosklearn.automl import AutoML
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.metrics import Scorer, accuracy, log_loss, precision, r2
from autosklearn.util.logging_ import PicklableClientLogger

import pytest
from pytest_cases import parametrize
from unittest.mock import patch


@parametrize(
    "dataset, metrics, task",
    [
        ("breast_cancer", [accuracy], BINARY_CLASSIFICATION),
        ("breast_cancer", [accuracy, log_loss], BINARY_CLASSIFICATION),
        ("wine", [accuracy], MULTICLASS_CLASSIFICATION),
        ("diabetes", [r2], REGRESSION),
    ],
)
def test_produces_correct_output(
    dataset: str,
    task: int,
    metrics: Sequence[Scorer],
    mock_logger: PicklableClientLogger,
    make_automl: Callable[..., AutoML],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> None:
    """
    Parameters
    ----------
    dataset: str
        The name of the dataset

    task : int
        The task type of the dataset

    metrics: Sequence[Scorer]
        Metric(s) to use, required as fit usually determines the metric to use

    Fixtures
    --------
    mock_logger: PickleableClientLogger
        A mock logger to use

    make_automl : Callable[..., AutoML]
        Factory to make an AutoML object

    make_sklearn_dataset : Callable[..., XYDataManager]
        Factory to get an sklearn dataset

    Expects
    -------
    * There should only be one output created with one dummy predictions
    * It should be named "1337_1_0.0" with {seed}_{num_run}_{budget}
    * It should produce predictions "predictions_ensemble_1337_1_0.0.npy"
    """
    seed = 1337
    automl = make_automl(metrics=metrics, seed=seed)
    automl._logger = mock_logger

    datamanager = make_sklearn_dataset(
        dataset,
        as_datamanager=True,
        task=task,
        feat_type="numerical",
    )
    automl._backend.save_datamanager(datamanager)
    automl._do_dummy_prediction()

    path = Path(automl._backend.get_runs_directory())
    run_paths = list(path.iterdir())
    assert len(run_paths) == 1

    dummy_run_path = run_paths[0]
    assert dummy_run_path.name == f"{seed}_1_0.0"

    predictions_path = dummy_run_path / f"predictions_ensemble_{seed}_1_0.0.npy"
    assert predictions_path.exists()


def test_runs_with_correct_args(
    mock_logger: PicklableClientLogger,
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
    make_automl: Callable[..., AutoML],
) -> None:
    """
    Fixtures
    --------
    mock_logger: PickleableClientLogger
        A mock logger to use

    make_sklearn_dataset : Callable[..., Tuple[np.ndarray, ...]]
        Factory to make dataset

    make_automl : Callable[..., AutoML]
        Factory to make automl

    Expects
    -------
    * The mock run should be called once with:
        * config = 1 (The always given number for the dummy)
        * cutoff = `automl._time_for_task` (the fulll time for the task)
    """
    dataset = "iris"
    task = MULTICLASS_CLASSIFICATION

    automl = make_automl(metrics=[accuracy])
    automl._logger = mock_logger

    datamanager = make_sklearn_dataset(
        dataset,
        as_datamanager=True,
        task=task,
        feat_type="numerical",
    )
    automl._backend.save_datamanager(datamanager)

    with patch("autosklearn.evaluation.ExecuteTaFuncWithQueue.run") as mck:
        mck.return_value = (StatusType.SUCCESS, None, None, {})
        automl._do_dummy_prediction()

    mck.assert_called_once_with(config=1, cutoff=automl._time_for_task)


def test_crash_due_to_memory_exception(
    mock_logger: PicklableClientLogger,
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
    make_automl: Callable[..., AutoML],
) -> None:
    """
    Fixtures
    --------
    mock_logger: PickleableClientLogger
        A mock logger to use

    make_sklearn_dataset : Callable[..., Tuple[np.ndarray, ...]]
        Factory to make dataset

    make_automl : Callable[..., AutoML]
        Factory to make automl

    Expects
    -------
    * The dummy prediction should raise when encoutering with StatusType.CRASHED
    * The error message should indicate it's a memory issue with `{'exitcode' -6}`
      encountered
    """
    dataset = "iris"
    task = MULTICLASS_CLASSIFICATION

    automl = make_automl(metrics=[accuracy])
    automl._logger = mock_logger

    datamanager = make_sklearn_dataset(
        dataset,
        as_datamanager=True,
        task=task,
        feat_type="numerical",
    )

    automl._backend.save_datamanager(datamanager)

    with patch("autosklearn.evaluation.ExecuteTaFuncWithQueue.run") as mck:
        mck.return_value = (StatusType.CRASHED, None, None, {"exitcode": -6})
        msg = "The error suggests that the provided memory limits are too tight."

        with pytest.raises(ValueError, match=msg):
            automl._do_dummy_prediction()


def test_raises_if_no_metric_set(make_automl: Callable[..., AutoML]) -> None:
    """
    Expects
    -------
    * raise if there was no metric set when calling `_do_dummy_prediction()`
    """
    automl = make_automl()
    with pytest.raises(ValueError, match="Metric/Metrics was/were not set"):
        automl._do_dummy_prediction()


def test_raises_invalid_metric(
    mock_logger: PicklableClientLogger,
    make_automl: Callable[..., AutoML],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> None:
    """
    Expects
    -------
    * Should raise an error if the given metric is not applicable to a given task type
    """
    # `precision` is not applicable to MULTICLASS_CLASSIFICATION
    dataset = "iris"
    task = MULTICLASS_CLASSIFICATION
    metrics = [accuracy, precision]

    automl = make_automl(metrics=metrics)
    automl._logger = mock_logger

    datamanager = make_sklearn_dataset(
        dataset,
        as_datamanager=True,
        task=task,
        feat_type="numerical",
    )
    automl._backend.save_datamanager(datamanager)

    with pytest.raises(
        ValueError,
        match="Are you sure precision is applicable for the given task type",
    ):
        automl._do_dummy_prediction()
