from __future__ import annotations

from typing import Tuple

import re
from pathlib import Path

import numpy as np

from autosklearn.util.disk import sizeof

RunID = Tuple[int, int, float]


class Run:
    """Class for storing information about a run used during ensemble building.

    Note
    ----
    This is for internal use by the EnsembleBuilder and not for general usage.
    """

    # For matching prediction files
    RE_MODEL_PREDICTION_FILE = (
        r"^predictions_ensemble_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*).npy$"
    )

    # For matching run directories
    RE_MODEL_DIR = r"^([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)$"

    def __init__(self, path: Path) -> None:
        """Creates a Run from a path pointing to the directory of a run

        Parameters
        ----------
        path: Path
            Expects something like /path/to/{seed}_{numrun}_{budget}
        """
        name = path.name
        seed, num_run, budget = name.split("_")

        self.dir = path
        self.seed = int(seed)
        self.num_run = int(num_run)
        self.budget = float(budget)

        # These are ordered based on preference
        self.losses: dict[str, float] = {}

        self._mem_usage: float | None = None

        # Items that will be delete when the run is saved back to file
        self._cache: dict[str, np.ndarray] = {}

        # The recorded time of ensemble/test predictions modified
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
        for kind in ["ensemble", "test"]:
            path = self.pred_path(kind)  # type: ignore
            if path.exists():
                self.recorded_mtimes[kind] = path.stat().st_mtime

    def has_predictions(self, kind: str = "ensemble") -> bool:
        """
        Parameters
        ----------
        kind: "ensemble" | "test" = "ensemble"
            The kind of predictions to query for

        Returns
        -------
        bool
            Whether this run has the kind of predictions queried for
        """
        return self.pred_path(kind).exists()

    def predictions(
        self,
        kind: str = "ensemble",
        precision: int | None = None,
    ) -> np.ndarray:
        """Load the predictions for this run

        Parameters
        ----------
        kind : "ensemble" | "test"
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

    def __getstate__(self) -> dict:
        """Remove the cache when pickling."""
        state = self.__dict__.copy()
        del state["_cache"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Reset state and instansiate blank cache."""
        self.__dict__.update(state)
        self._cache = {}

    @property
    def id(self) -> RunID:
        """Get the three components of it's id"""
        return self.seed, self.num_run, self.budget

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Run(id={self.id}, losses={self.losses})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and other.id == self.id

    @staticmethod
    def valid(path: Path) -> bool:
        """
        Parameters
        ----------
        path: Path
            The path to check

        Returns
        -------
        bool
            Whether the path is a valid run dir
        """
        return re.match(Run.RE_MODEL_DIR, path.name) is not None
