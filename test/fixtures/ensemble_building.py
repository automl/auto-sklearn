from __future__ import annotations

from typing import Callable

import math
from pathlib import Path

import numpy as np

from autosklearn.ensemble_building import Run

from pytest_cases import fixture

from test.conftest import DEFAULT_SEED


@fixture
def make_run(tmp_path: Path) -> Callable[..., Run]:
    def _make(
        id: int | None = 2,
        seed: int = DEFAULT_SEED,
        budget: float = 0.0,
        loss: float | None = None,
        model_size: int | None = None,
        predictions: list[str] | dict[str, np.ndarray] | None = None,
    ) -> Run:
        model_id = f"{seed}_{id}_{budget}"
        dir = tmp_path / model_id

        if not dir.exists():
            dir.mkdir()

        # Populate if None
        if predictions is None:
            predictions = ["ensemble", "valid", "test"]

        # Convert to dict
        if isinstance(predictions, list):
            dummy = np.asarray([[0]])
            predictions = {kind: dummy for kind in predictions}

        # Write them
        if isinstance(predictions, dict):
            for kind, val in predictions.items():
                fname = f"predictions_{kind}_{seed}_{id}_{budget}.npy"
                with (dir / fname).open("wb") as f:
                    np.save(f, val)

        run = Run(dir)

        if loss is not None:
            run.loss = loss

        # MB
        if model_size is not None:
            n_bytes = int(model_size * math.pow(1024, 2))
            model_path = dir / f"{seed}.{id}.{budget}.model"
            with model_path.open("wb") as f:
                f.write(bytearray(n_bytes))

        return run

    return _make
