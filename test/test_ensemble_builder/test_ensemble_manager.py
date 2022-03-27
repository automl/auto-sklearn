from typing import Callable

import time
from pathlib import Path

import dask.distributed
import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.ensemble_builder import EnsembleBuilderManager
from autosklearn.metrics import roc_auc

from pytest_cases import parametrize_with_cases

import test.test_ensemble_builder.cases as cases


@parametrize_with_cases("ensemble_backend", cases=cases)
def test_ensemble_builder_nbest_remembered(
    ensemble_backend: Backend,
    make_dask_client: Callable[..., [dask.distributed.Client]],
) -> None:
    """
    Parameters
    ----------
    ensemble_backend: Backend
        The backend to use, relies on the 3 setup models

    Fixtures
    --------
    make_dask_client: (...) -> Client
        Make a dask client

    Expects
    -------
    * The read_preds file should not be created
    * The ensemble_nbest should be remembered and reduced between runs
    TODO Note sure why there would be a reduction and how these numbers were made

    Last Note
    ---------
    "Makes sure ensemble builder returns the size of the ensemble that pynisher allowed
    This way, we can remember it and not waste more time trying big ensemble sizes"
    """
    dask_client = make_dask_client(n_workers=1)

    manager = EnsembleBuilderManager(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        backend=ensemble_backend,
        dataset_name="Test",
        task=BINARY_CLASSIFICATION,
        metric=roc_auc,
        ensemble_size=50,
        ensemble_nbest=10,
        max_models_on_disc=None,
        seed=0,
        precision=32,
        read_at_most=np.inf,
        ensemble_memory_limit=1000,
        random_state=0,
        max_iterations=None,
    )

    filepath = Path(ensemble_backend.internals_directory) / "ensemble_read_preds.pkl"

    manager.build_ensemble(dask_client, unit_test=True)
    future = manager.futures[0]
    dask.distributed.wait([future])  # wait for the ensemble process to finish

    assert future.result() == ([], 5, None, None, None)

    assert not filepath.exists()

    manager.build_ensemble(dask_client, unit_test=True)
    future = manager.futures[0]
    dask.distributed.wait([future])  # wait for the ensemble process to finish

    assert not filepath.exists()
    assert future.result() == ([], 2, None, None, None)
