from __future__ import annotations

from typing import Any, Tuple
from typing_extensions import Literal

from pathlib import Path

import numpy as np

from autosklearn.util.disk import sizeof

RunID = Tuple[int, int, float]


class Run:
    """Class for storing information about a run used during ensemble building"""

    def __init__(self, path: Path) -> None:
        """Creates a Run from a path point to the directory of a run

        Parameters
        ----------
        path: Path
            Expects something like /path/to/{seed}_{numrun}_{budget}

        Returns
        -------
        Run
            The run object generated from the directory
        """
        name = path.name
        seed, num_run, budget = name.split("_")

        self.dir = path
        self.seed = int(seed)
        self.num_run = int(num_run)
        self.budget = float(budget)

        self.loss: float | None = None
        self._mem_usage: float | None = None

        # Items that will be delete when the run is saved back to file
        self._cache: dict[str, Any] = {}

        # The recorded time of ensemble/test/valid predictions modified
        self.recorded_mtimes: dict[str, float] = {}
        self.record_modified_times()

    @property
    def mem_usage(self) -> float:
        """The memory usage of this run based on it's directory"""
        if self._mem_usage is None:
            self._mem_usage = round(sizeof(self.dir, unit="MB"), 2)

        return self._mem_usage

    def is_dummy(self) -> bool:
        """Whether this run is a dummy run or not"""
        return self.num_run == 1

    def pred_modified(self, kind: Literal["ensemble", "valid", "test"]) -> bool:
        """Query for when the ens file was last modified"""
        if kind not in self.recorded_mtimes:
            raise ValueError(f"Run has no recorded time for {kind}: {self}")

        recorded = self.recorded_mtimes[kind]
        last = self.pred_path(kind).stat().st_mtime

        return recorded == last

    def pred_path(self, kind: Literal["ensemble", "valid", "test"]) -> Path:
        """Get the path to certain predictions"""
        fname = f"predictions_{kind}_{self.seed}_{self.num_run}_{self.budget}.npy"
        return self.dir / fname

    def record_modified_times(self) -> None:
        """Records the last time each prediction file type was modified, if it exists"""
        self.recorded_mtimes = {}
        for kind in ["ensemble", "valid", "test"]:
            path = self.pred_path(kind)  # type: ignore
            if path.exists():
                self.recorded_mtimes[kind] = path.stat().st_mtime

    def predictions(
        self,
        kind: Literal["ensemble", "valid", "test"],
        precision: int | None = None,
    ) -> Path:
        """Load the predictions for this run

        Parameters
        ----------
        kind : Literal["ensemble", "valid", "test"]
            The kind of predictions to load

        precisions : type | None = None
            What kind of precision reduction to apply

        Returns
        -------
        np.ndarray
            The loaded predictions
        """
        key = f"predictions_{kind}"
        if key in self._cache:
            return self._cache[key]

        path = self.pred_path(kind)

        with path.open("rb") as f:
            # TODO: We should probably remove this requirement. I'm not sure why model
            # predictions are being saved as pickled
            predictions = np.load(f, allow_pickle=True)

        if precision:
            dtypes: dict[int, type] = {16: np.float16, 32: np.float32, 64: np.float64}
            dtype = dtypes.get(precision, None)

            if dtype is not None:
                predictions = predictions.astype(dtype=dtype, copy=False)

        self._cache[key] = predictions
        return predictions

    def unload_cache(self) -> None:
        """Removes the cache from this object"""
        self._cache = {}

    @property
    def id(self) -> RunID:
        """Get the three components of it's id"""
        return self.seed, self.num_run, self.budget

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and other.id == self.id
