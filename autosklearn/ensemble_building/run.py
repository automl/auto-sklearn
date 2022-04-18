from __future__ import annotations

from typing import Tuple

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

        self.loss: float = np.inf
        self._mem_usage: float | None = None

        # Items that will be delete when the run is saved back to file
        self._cache: dict[str, np.ndarray] = {}

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

    def was_modified(self) -> bool:
        """Query for when the ens file was last modified"""
        recorded = self.recorded_mtimes.get("ensemble")
        last = self.pred_path().stat().st_mtime
        return recorded != last

    def pred_path(self, kind: str = "ensemble") -> Path:
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
        kind: str = "ensemble",
        precision: int | None = None,
    ) -> np.ndarray:
        """Load the predictions for this run

        Parameters
        ----------
        kind : "ensemble" | "test" | "valid"
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

        if not path.exists():
            raise RuntimeError(f"No predictions for {kind}")

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
        """Removes the cache from this object

        We could also enforce that nothing gets pickled to disk with __getstate__
        but this is simpler and shows expliciyt behaviour in caller code.
        """
        self._cache = {}

    @property
    def id(self) -> RunID:
        """Get the three components of it's id"""
        return self.seed, self.num_run, self.budget

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Run(id={self.id}, loss={self.loss})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and other.id == self.id
