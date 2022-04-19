from __future__ import annotations

from typing import Any, Callable

import math
import pickle
import sys
from pathlib import Path

import numpy as np

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
        model_size: int | None = None,
        mem_usage: float | None = None,
        predictions: str | list[str] | dict[str, np.ndarray] | None = "ensemble",
    ) -> Run:
        if dummy:
            assert id is None
            id = 1
            loss = loss if loss is not None else 50_000

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
            run.loss = loss

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
        previous_candidates: list[Run] | None = None,
        backend: Backend | None = None,
        dataset_name: str = "TEST",
        task_type: int = BINARY_CLASSIFICATION,
        metric: Scorer = accuracy,
        **kwargs: Any,
    ) -> EnsembleBuilder:

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
                backend.save_targets_ensemble(datamanager.data["Y_train"])

        builder = EnsembleBuilder(
            backend=backend,
            dataset_name=dataset_name,
            task_type=task_type,
            metric=metric,
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
                backend.save_targets_ensemble(datamanager.data["Y_train"])

        return EnsembleBuilderManager(
            backend=backend,
            dataset_name=dataset_name,
            task=task,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )

    return _make
