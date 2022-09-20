from __future__ import annotations

from typing import Any, Callable

import copy
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.ensemble_building import EnsembleBuilder, EnsembleBuilderManager, Run
from autosklearn.metrics import Scorer, accuracy

from pytest_cases import fixture

from test.conftest import DEFAULT_SEED


@fixture
def make_run(tmp_path: Path) -> Callable[..., Run]:
    def _make(
        id: int | None = None,
        dummy: bool = False,
        backend: Backend | None = None,
        seed: int = DEFAULT_SEED,
        modified: bool = False,
        budget: float = 0.0,
        loss: float | None = None,
        losses: dict[str, float] | list[float] | None = None,
        model_size: int | None = None,
        mem_usage: float | None = None,
        predictions: str | list[str] | dict[str, np.ndarray] | None = "ensemble",
    ) -> Run:
        if loss is not None and losses is not None:
            raise ValueError("Can only specify either `loss` or `losses`")

        if isinstance(loss, dict):
            raise ValueError("Please use `losses` for dict of losses")

        if dummy:
            assert id is None
            id = 1
            if loss is None and losses is None:
                losses = {"metric_0": 50_000}

        if id is None:
            id = np.random.randint(sys.maxsize)

        model_id = f"{seed}_{id}_{budget}"

        # Use this backend to set things up
        if backend is not None:
            runsdir = Path(backend.get_runs_directory())
        else:
            runsdir = tmp_path

        dir = runsdir / model_id

        if not dir.exists():
            dir.mkdir()

        # Populate if None
        if isinstance(predictions, str):
            predictions = [predictions]

        # Convert to dict
        if isinstance(predictions, list):
            preds = np.asarray([[0]])
            predictions = {kind: preds for kind in predictions}

        # Write them
        if isinstance(predictions, dict):
            for kind, val in predictions.items():
                fname = f"predictions_{kind}_{seed}_{id}_{budget}.npy"
                with (dir / fname).open("wb") as f:
                    np.save(f, val)

        run = Run(dir)

        if modified:
            assert predictions is not None, "Can only modify if predictions"
            for k, v in run.recorded_mtimes.items():
                run.recorded_mtimes[k] = v + 1e-4

        if loss is not None:
            losses = [loss]

        if isinstance(losses, list):
            losses = {f"metric_{i}": loss for i, loss in enumerate(losses)}

        if isinstance(losses, dict):
            run.losses = losses

        if mem_usage is not None:
            run._mem_usage = mem_usage

        # MB
        if model_size is not None:
            n_bytes = int(model_size * math.pow(1024, 2))
            model_path = dir / f"{seed}.{id}.{budget}.model"
            with model_path.open("wb") as f:
                f.write(bytearray(n_bytes))

        return run

    return _make


@fixture
def make_ensemble_builder(
    make_backend: Callable[..., Backend],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> Callable[..., EnsembleBuilder]:
    def _make(
        *,
        automl: AutoML | None = None,
        previous_candidates: list[Run] | None = None,
        backend: Backend | None = None,
        dataset_name: str = "TEST",
        task_type: int = BINARY_CLASSIFICATION,
        metrics: Scorer = copy.deepcopy([accuracy]),
        **kwargs: Any,
    ) -> EnsembleBuilder:

        if automl:
            backend = automl._backend
            dataset_name = automl._dataset_name
            task_type = automl._task
            metrics = automl._metrics
            kwargs = {
                "ensemble_class": automl._ensemble_class,
                "ensemble_kwargs": automl._ensemble_kwargs,
                "ensemble_nbest": automl._ensemble_nbest,
                "max_models_on_disc": automl._max_models_on_disc,
                "precision": automl.precision,
                "read_at_most": automl._read_at_most,
                "memory_limit": automl._memory_limit,
                "logger_port": automl._logger_port,
            }

        if backend is None:
            backend = make_backend()

        # If there's no datamanager, just try populate it with some generic one,
        if not Path(backend._get_datamanager_pickle_filename()).exists():
            datamanager = make_sklearn_dataset(
                name="breast_cancer",
                task=BINARY_CLASSIFICATION,
                feat_type="numerical",  # They're all numerical
                as_datamanager=True,
            )
            backend.save_datamanager(datamanager)

            # Annoyingly, some places use datamanger, some places use the file
            # Hence, we take the y_train of the datamanager and use that as the
            # the targets
            if "Y_train" in datamanager.data:
                backend.save_additional_data(
                    datamanager.data["Y_train"], what="targets_ensemble"
                )
            if "X_train" in datamanager.data:
                backend.save_additional_data(
                    datamanager.data["X_train"], what="input_ensemble"
                )

        builder = EnsembleBuilder(
            backend=backend,
            dataset_name=dataset_name,
            task_type=task_type,
            metrics=metrics,
            **kwargs,
        )

        if previous_candidates is not None:
            with builder.previous_candidates_path.open("wb") as f:
                pickle.dump({run.id: run for run in previous_candidates}, f)

        return builder

    return _make


@fixture
def make_ensemble_builder_manager(
    make_backend: Callable[..., Backend],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> Callable[..., EnsembleBuilderManager]:
    """Use `make_run` to create runs for this manager

    .. code:: python

        def test_x(make_run, make_ensemble_builder_manager):
            manager = make_ensemble_builder(...)

            # Will use the backend to place runs correctly
            runs = make_run(predictions={"ensemble": ...}, backend=manager.backend)

            # ... test stuff


    """

    def _make(
        *,
        backend: Backend | None = None,
        dataset_name: str = "TEST",
        task: int = BINARY_CLASSIFICATION,
        metric: Scorer = accuracy,
        random_state: int | np.random.RandomState | None = DEFAULT_SEED,
        **kwargs: Any,
    ) -> EnsembleBuilderManager:
        if backend is None:
            backend = make_backend()

        if not Path(backend._get_datamanager_pickle_filename()).exists():
            datamanager = make_sklearn_dataset(
                name="breast_cancer",
                task=BINARY_CLASSIFICATION,
                feat_type="numerical",  # They're all numerical
                as_datamanager=True,
            )
            backend.save_datamanager(datamanager)

            # Annoyingly, some places use datamanger, some places use the file
            # Hence, we take the y_train of the datamanager and use that as the
            # the targets
            if "Y_train" in datamanager.data:
                backend.save_additional_data(
                    datamanager.data["Y_train"],
                    what="targets_ensemble",
                )

        return EnsembleBuilderManager(
            backend=backend,
            dataset_name=dataset_name,
            task=task,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )

    return _make
