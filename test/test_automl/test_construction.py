"""Test things related to only constructing an AutoML instance"""
from typing import Any, Dict, Optional, Union

from autosklearn.automl import AutoML
from autosklearn.util.dask import LocalDask
from autosklearn.util.data import default_dataset_compression_arg
from autosklearn.util.single_thread_client import SingleThreadedClient

import pytest
from pytest_cases import parametrize


@parametrize("disable_evaluator_output", [("hello", "world"), ("model", "other")])
def test_invalid_disable_eval_output_options(disable_evaluator_output: Any) -> None:
    """
    Parameters
    ----------
    disable_evaluator_output : Iterable[str]
        An iterator of invalid options

    Expects
    -------
    * Should raise an error about invalid options
    """
    with pytest.raises(ValueError, match="Unknown arg"):
        AutoML(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            disable_evaluator_output=disable_evaluator_output,
        )


@parametrize(
    "dataset_compression, expected",
    [
        (False, None),
        (True, default_dataset_compression_arg),
        (
            {"memory_allocation": 0.2},
            {**default_dataset_compression_arg, **{"memory_allocation": 0.2}},
        ),
    ],
)
def test_param_dataset_compression_args(
    dataset_compression: Union[bool, Dict],
    expected: Optional[Dict],
) -> None:
    """
    Parameters
    ----------
    dataset_compression: Union[bool, Dict]
        The dataset_compression arg used

    expected: Optional[Dict]
        The expected internal variable setting

    Expects
    -------
    * Setting the compression arg should result in the expected value
        * False -> None, No dataset compression
        * True -> default, The default settings
        * dict -> default updated, The default should be updated with the args used
    """
    auto = AutoML(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        dataset_compression=dataset_compression,
    )
    assert auto._dataset_compression == expected


def test_single_job_and_no_dask_client_sets_correct_multiprocessing_context() -> None:
    """
    Expects
    -------
    * With n_jobs set to 1 and no dask client, we default to a SingleThreadedClient
    with a "fork" _multiprocessing_context
    """
    n_jobs = 1
    dask_client = None

    automl = AutoML(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        n_jobs=n_jobs,
        dask_client=dask_client,
    )

    assert automl._multiprocessing_context == "fork"
    assert automl._n_jobs == 1
    assert isinstance(automl._dask, LocalDask)

    with automl._dask as client:
        assert isinstance(client, SingleThreadedClient)
