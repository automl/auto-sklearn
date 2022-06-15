"""Test specific ways of calling `fit` of AutoML"""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from dask.distributed import Client
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

from autosklearn.automl import AutoML
from autosklearn.constants import MULTICLASS_CLASSIFICATION

from pytest_cases import parametrize


@parametrize("dataset, task, bounds", [("iris", MULTICLASS_CLASSIFICATION, (0.8, 1.0))])
def test_fit_roar(
    dataset: str,
    task: int,
    bounds: Tuple[float, float],
    dask_client_single_worker: Client,
    make_automl: Callable[..., AutoML],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> None:
    """
    Parameters
    ----------
    dataset : str
        The name of the dataset

    task : int
        The task type of the dataset

    bounds : Tuple[float, float]
        The bounds the final score should be in, (lowest, upper)

    Fixtures
    --------
    make_automl : Callable[..., AutoML]
        Factory for making an AutoML instance

    make_sklearn_dataset : Callable[..., Tuple[np.ndarray, ...]]
        Factory for getting a dataset

    Expects
    -------
    * Should fit without a problem using a different smac object
    """

    def get_roar_object_callback(
        scenario_dict: Dict,
        seed: Optional[Union[int, np.random.RandomState]],
        ta: Callable,
        ta_kwargs: Dict,
        dask_client: Client,
        n_jobs: int,
        **kwargs: Any,
    ) -> ROAR:
        """Random online adaptive racing.

        http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf
        """
        scenario = Scenario(scenario_dict)
        return ROAR(
            run_id=seed,
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            dask_client=dask_client,
            n_jobs=n_jobs,
        )

    X_train, Y_train, X_test, Y_test = make_sklearn_dataset(dataset)
    automl = make_automl(
        initial_configurations_via_metalearning=0,
        get_smac_object_callback=get_roar_object_callback,
        dask_client=dask_client_single_worker,
    )

    automl.fit(X_train, Y_train, task=task)

    score = automl.score(X_test, Y_test)
    assert score > 0.8
