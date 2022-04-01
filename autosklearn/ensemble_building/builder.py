from __future__ import annotations

from typing import Any, Sequence

import logging.handlers
import multiprocessing
import numbers
import os
import pickle
import shutil
import time
import traceback
import zlib
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path

import numpy as np
import pandas as pd
import pynisher
from typing_extensions import Literal

from autosklearn.automl_common.common.ensemble_building.abstract_ensemble import (  # noqa: E501
    AbstractEnsemble,
)
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import Scorer, calculate_loss, calculate_score
from autosklearn.util.disk import sizeof
from autosklearn.util.functional import bound, findwhere, intersection, itersplit
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2


RunID = tuple[int, int, float]


@dataclass
class Run:
    """Dataclass for storing information about a run"""

    seed: int
    num_run: int
    ens_file: str
    dir: Path
    budget: float = 0.0
    loss: float | None = None
    _mem_usage: int | None = None
    # The recorded time of ensemble/test/valid predictions modified
    recorded_mtimes: dict[str, float] = 0
    # Lazy keys so far:
    # 0 - not loaded
    # 1 - loaded and in memory
    # 2 - loaded but dropped again
    # 3 - deleted from disk due to space constraints
    loaded: int = 0

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
        if self.recorded_mtimes is None:
            raise RuntimeError("No times were recorded, use `record_modified_times`")

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
        for kind in ["ensemble", "valid", "test"]:
            path = self.pred_path(kind)
            if path.exists():
                self.recorded_mtimes[kind] = path.stat().st_mtime()

    def predictions(
        self,
        kind: Literal["ensemble", "valid", "test"],
        precision: type | None = None,
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
        path = self.pred_path(kind)

        with path.open("rb") as f:
            # TODO: We should probably remove this requirement. I'm not sure why model
            # predictions are being saved as pickled
            predictions = np.load(f, allow_pickle=True)

        dtypes = {16: np.float16, 32: np.float32, 64: np.float64}
        dtype = dtypes.get(precision, predictions.dtype)
        predictions = predictions.astype(dtype=dtype, copy=False)

        return predictions

    @property
    def id(self) -> RunID:
        """Get the three components of it's id"""
        return self.seed, self.num_run, self.budget

    def __str__(self) -> str:
        return f"{self.seed}_{self.num_run}_{self.budget}"

    @staticmethod
    def from_dir(dir: Path) -> Run:
        """Creates a Run from a path point to the directory of a run

        Parameters
        ----------
        dir: Path
            Expects something like /path/to/{seed}_{numrun}_budget

        Returns
        -------
        Run
            The run object generated from the directory
        """
        name = dir.name
        seed, num_run, budget = name.split("_")
        run = Run(seed=seed, num_run=num_run, budget=budget, dir=dir)
        run.record_modified_times()
        return run


class EnsembleBuilder:
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        metric: Scorer,
        ensemble_size: int = 10,
        ensemble_nbest: int | float = 100,
        max_models_on_disc: int | float | None = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: int | None = 1024,
        read_at_most: int | None = 5,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        random_state: int | np.random.RandomState | None = None,
    ):
        """
        Parameters
        ----------
        backend: Backend
            backend to write and read files

        dataset_name: str
            name of dataset

        task_type: int
            type of ML task

        metric: str
            name of metric to compute the loss of the given predictions

        ensemble_size: int = 10
            maximal size of ensemble (passed to autosklearn.ensemble.ensemble_selection)

        ensemble_nbest: int | float = 100

            * int: consider only the n best prediction (> 0)

            * float: consider only this fraction of the best, between (0, 1)

            Both with respect to the validation predictions.
            If performance_range_threshold > 0, might return less models

        max_models_on_disc: int | float | None = 100
           Defines the maximum number of models that are kept in the disc.
           It defines an upper bound on the models that can be used in the ensemble.

           * int: and dictates the max number of models to keep. (>= 1)

           * float: it will be interpreted as the max megabytes allowed of disc space.
           If the number of ensemble candidates require more disc space than
           this float value, the worst models are deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.

           * None: the feature is disabled.

        performance_range_threshold: float = 0
            Will at most return the minimum between ensemble_nbest models,
            and max_models_on_disc. Might return less

            Keep only models that are better than:

                dummy + (best - dummy) * performance_range_threshold

            E.g dummy=2, best=4, thresh=0.5 --> only consider models with loss > 3

        seed: int = 1
            random seed that is used as part of the filename

        precision: int [16 | 32 | 64 | 128] = 32
            precision of floats to read the predictions

        memory_limit: int | None = 1024
            memory limit in mb. If ``None``, no memory limit is enforced.

        read_at_most: int | None = 5
            read at most n new prediction files in each iteration

        logger_port: int = DEFAULT_TCP_LOGGING_PORT
            port that receives logging records

        random_state: int | RandomState | None = None
            An int or RandomState object used for generating the ensemble.
        """
        if isinstance(ensemble_nbest, int) and ensemble_nbest < 1:
            raise ValueError(f"int ensemble_nbest ({ensemble_nbest}) must be (>1)")

        if isinstance(ensemble_nbest, float) and not (0 <= ensemble_nbest <= 1):
            raise ValueError(f"float ensemble_nbest ({ensemble_nbest}) not in (0,1)")

        if max_models_on_disc is not None and max_models_on_disc < 0:
            raise ValueError("max_models_on_disc must be positive or None")

        if read_at_most is not None and read_at_most < 1:
            raise ValueError("Read at most must be greater than 1")

        # Setup the logger
        self.logger = get_named_client_logger(name="EnsembleBuilder", port=logger_port)
        self.logger_port = logger_port

        # Log the behaviour
        if ensemble_nbest == 1:
            t = type(ensemble_nbest)
            self.logger.debug(f"Using behaviour when {t} for {ensemble_nbest}:{t}")

        self.seed = seed
        self.metric = metric
        self.backend = backend
        self.precision = precision
        self.task_type = task_type
        self.memory_limit = memory_limit
        self.read_at_most = read_at_most
        self.random_state = random_state
        self.dataset_name = dataset_name
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.performance_range_threshold = performance_range_threshold

        # The starting time of the procedure
        self.start_time = 0

        # Track the ensemble performance
        self.ensemble_history = []

        # Keep running knowledge of its validation performance
        self.validation_performance_ = np.inf

        # Data we may need
        datamanager = self.backend.load_datamanager()
        self._y_valid: np.ndarray | None = datamanager.data.get("Y_valid", None)
        self._y_test: np.ndarray | None = datamanager.data.get("Y_test", None)
        self._y_ensemble: np.ndarray | None = None

        # max_resident_models keeps the maximum number of models in disc
        # Calculated during `main`
        self.max_resident_models: int | None = None

        # Cached items, loaded by properties
        # Check the corresponing properties for descriptions
        self._run_prediction_paths: list[str] | None = None
        self._run_predictions: dict[str, dict[int, np.ndarray]] | None = None
        self._last_hash: str | None = None
        self._runs: dict[str, Run] | None = None

    @property
    def run_predictions_path(self) -> Path:
        """Path to the cached predictions we store between runs"""
        return Path(self.backend.internals_directory) / "ensemble_read_preds.pkl"

    @property
    def runs_path(self) -> Path:
        """Path to the cached losses we store between runs"""
        return Path(self.backend.internals_directory) / "ensemble_read_losses.pkl"

    @property
    def run_predictions(self) -> dict[str, dict[int, np.ndarray]]:
        """Get the cached predictions from previous runs
        {
            "file_name": {
                Y_ENSEMBLE: np.ndarray
                Y_VALID: np.ndarray
                Y_TEST: np.ndarray
            }
        }
        """
        if self._run_predictions is None:
            self._run_predictions = {}
            self._last_hash = ""

            path = self.run_predictions_path
            if path.exists():
                with path.open("rb") as memory:
                    self._run_predictions, self._last_hash = pickle.load(memory)

        return self._run_predictions

    @property
    def last_hash(self) -> str:
        """Get the last hash associated with the run predictions"""
        if self._last_hash is None:
            self._run_predictions = {}
            self._last_hash = ""

            path = self.run_predictions_path
            if path.exists():
                with path.open("rb") as memory:
                    self._run_predictions, self._last_hash = pickle.load(memory)

        return self._last_hash

    @property
    def runs(self) -> list[Run]:
        """Get the cached information from previous runs"""
        if self._runs is None:
            # First read in all the runs on disk
            runs_dir = Path(self.backend.get_runs_directory())
            all_runs = [Run.from_dir(dir) for dir in runs_dir.iterdir()]

            # Next, get the info about runs from last EnsembleBulder run, if any
            loaded_runs: dict[RunID, Run] = {}
            if self.runs_path.exists():
                with self.runs_path.open("rb") as memory:
                    loaded_runs = pickle.load(memory)

            # Update any run that was loaded but we didn't have previously
            for run in all_runs:
                if run.id not in loaded_runs:
                    loaded_runs[run.id] = run

            self._runs = loaded_runs

        return list(self._runs.values())

    def targets(self, kind: Literal["ensemble", "valid", "test"]) -> np.ndarray | None:
        """The ensemble targets used for training the ensemble

        It will attempt to load and cache them in memory but
        return None if it can't.

        Returns
        -------
        np.ndarray | None
            The ensemble targets, if they can be loaded
        """
        if kind == "ensemble" and self._y_ensemble is None:
            if os.path.exists(self.backend._get_targets_ensemble_filename()):
                self._y_ensemble = self.backend.load_targets_ensemble()
            return self._y_ensemble

        elif kind == "valid":
            return self._y_valid

        elif kind == "test":
            return self._y_test

        else:
            raise NotImplementedError()

    def run(
        self,
        iteration: int,
        pynisher_context: str | None = None,
        time_left: float | None = None,
        end_at: float | None = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        int,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Run the ensemble building process

        Parameters
        ----------
        iteration : int
            What iteration to associate with this run

        pynisher_context : str | None = None
            The pynisher context to run in. If None, defaults to
            multiprocessing.get_context(None)

        time_left : float | None = None
            How much time should be left for this run. Either this or `end_at` must
            be provided.

        end_at : float | None = Non
            When this run should end. Either this or `time_left` must be provided.

        time_buffer : int = 5
            How much extra time to add as a buffer to this run. This means there is
            always some amount of time to do something useful.

        return_predictions : bool = False
            Whether run should also return predictions

        Returns
        -------
        (ensemble_history, nbest, train_preds, valid_preds, test_preds)
        """
        if time_left is None and end_at is None:
            raise ValueError("Must provide either time_left or end_at.")

        elif time_left is not None and end_at is not None:
            raise ValueError("Cannot provide both time_left and end_at.")

        if not self.logger:
            self.logger = get_named_client_logger(
                name="EnsembleBuilder",
                port=self.logger_port,
            )

        process_start_time = time.time()
        while True:

            if time_left is not None:
                time_elapsed = time.time() - process_start_time
                time_left -= time_elapsed
            else:
                current_time = time.time()
                if current_time > end_at:
                    break
                else:
                    time_left = end_at - current_time

            wall_time_in_s = int(time_left - time_buffer)
            if wall_time_in_s < 1:
                break

            context = multiprocessing.get_context(pynisher_context)
            preload_modules(context)

            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                mem_in_mb=self.memory_limit,
                logger=self.logger,
                context=context,
            )(self.main)
            safe_ensemble_script(time_left, iteration, return_predictions)
            if safe_ensemble_script.exit_status is pynisher.MemorylimitException:
                # if ensemble script died because of memory error,
                # reduce nbest to reduce memory consumption and try it again

                # ATTENTION: main will start from scratch;
                # all data structures are empty again
                try:
                    self.run_predictions_path.unlink()
                except:  # noqa E722
                    pass

                if (
                    isinstance(self.ensemble_nbest, numbers.Integral)
                    and self.ensemble_nbest <= 1
                ):
                    if self.read_at_most == 1:
                        self.logger.error(
                            "Memory Exception -- Unable to further reduce the number"
                            " of ensemble members and can no further limit the number"
                            " of ensemble members loaded per iteration, please restart"
                            " Auto-sklearn with a higher value for the argument"
                            f" `memory_limit` (current limit is {self.memory_limit}MB)."
                            " The ensemble builder will keep running to delete files"
                            " from disk in case this was enabled.",
                        )
                        self.ensemble_nbest = 0
                    else:
                        self.read_at_most = 1
                        self.logger.warning(
                            "Memory Exception -- Unable to further reduce the number of"
                            " ensemble members. Now reducing the number of predictions"
                            " per call to read at most to 1."
                        )
                else:
                    if isinstance(self.ensemble_nbest, numbers.Integral):
                        self.ensemble_nbest = max(1, int(self.ensemble_nbest / 2))
                    else:
                        self.ensemble_nbest = self.ensemble_nbest / 2
                    self.logger.warning(
                        "Memory Exception -- restart with "
                        "less ensemble_nbest: %d" % self.ensemble_nbest
                    )
                    return [], self.ensemble_nbest, None, None, None
            else:
                return safe_ensemble_script.result

        return [], self.ensemble_nbest, None, None, None

    def main(
        self,
        time_left: float,
        iteration: int,
        return_predictions: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        int,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Run the main loop of ensemble building

        Parameters
        ----------
        time_left : float
            How much time is left for this run

        iteration : int
            The iteration of this run

        return_predictions : bool = False
            Whether to return predictions or not

        Returns
        -------
        (ensemble_history, nbest, train_preds, valid_preds, test_preds)
        """
        # Pynisher jobs inside dask 'forget' the logger configuration.
        # So we have to set it up accordingly
        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, valid_pred, test_pred = None, None, None

        used_time = time.time() - self.start_time
        left_for_iter = time_left - used_time
        self.logger.debug(f"Starting iteration {iteration}, time left: {left_for_iter}")

        # Can't load data, exit early
        if not os.path.exists(self.backend._get_targets_ensemble_filename()):
            self.logger.debug(f"No targets for ensemble: {traceback.format_exc()}")
            return self.ensemble_history, self.ensemble_nbest, None, None, None

        # Get our runs
        runs = self.runs

        # No runs found, exit early
        if len(self.runs) == 0:
            self.logger.debug("Found no predictions on ensemble data set")
            return self.ensemble_history, self.ensemble_nbest, None, None, None

        # We filter out all runs that don't have any predictions for the ensemble
        with_predictions, without_predictions = itersplit(
            runs, func=lambda r: r.pred_path("ensemble").exists()
        )

        if len(without_predictions) > 0:
            self.logger.warn(f"Have no ensemble predictions for {without_predictions}")

        runs = with_predictions

        # Calculating losses
        #
        #   We need to calculate the loss of runs for which we have not done so yet.
        #   To do so, we first filter out runs that already have a loss
        #   and have not had their predictions modified.
        #
        #   We then compute the losses for the runs remaining, sorted by their
        #   last-modified time, such that oldest are computed first. We only compute
        #   `self.read_at_most` of them, if specified, to ensure we don't spend too much
        #   time reading and computing losses.
        #
        # Filter runs that need their losses computed
        runs_to_compute_loss = []
        for run in runs:
            if run.loss is None or run.loss == np.inf:
                runs_to_compute_loss.append(run)

            elif run.loss is not None and run.pred_modified("ensemble"):
                self.logger.debug(f"{run.id} had its predictions modified?")
                runs_to_compute_loss.append(run)

        # Sort by last modified
        by_last_modified = lambda r: r.record_mtimes["ensemble"]
        runs_to_compute_loss = sorted(runs_to_compute_loss, key=by_last_modified)

        # Limit them if needed
        if self.read_at_most is not None:
            runs_to_compute_loss = runs_to_compute_loss[: self.read_at_most]

        # Calculate their losses
        ensemble_targets = self.targets("ensemble")
        for run in runs_to_compute_loss:
            loss = self.run_loss(run, targets=ensemble_targets, kind="ensemble")
            run.loaded = 2
            run.loss = loss

        n_read_total = sum(run.loaded > 0 for run in runs)
        self.logger.debug(
            f"Done reading {len(runs_to_compute_loss)} new prediction files."
            f"Loaded {n_read_total} predictions in total."
        )

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_nbest()
        if not candidate_models:  # no candidates yet
            if return_predictions:
                return (
                    self.ensemble_history,
                    self.ensemble_nbest,
                    train_pred,
                    valid_pred,
                    test_pred,
                )
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None, None

        # populates predictions in self.run_predictions
        # reduces selected models if file reading failed
        n_sel_valid, n_sel_test = self.get_valid_test_preds(
            selected_keys=candidate_models
        )

        # Get a set representation of them as we will begin doing intersections
        candidates_set = set(candidate_models)
        valid_set = set(n_sel_valid)
        test_set = set(n_sel_test)

        # Both n_sel_* have entries, but there is no overlap, this is critical
        if len(test_set) > 0 and len(valid_set) > 0 and len(valid_set & test_set) == 0:
            self.logger.error("n_sel_valid and n_sel_test not empty but do not overlap")
            if return_predictions:
                return (
                    self.ensemble_history,
                    self.ensemble_nbest,
                    train_pred,
                    valid_pred,
                    test_pred,
                )
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None, None

        intersect = intersection(candidates_set, valid_set, test_set)
        if len(intersect) > 0:
            candidate_models = sorted(list(intersect))
            n_sel_test = candidate_models
            n_sel_valid = candidate_models

        elif len(candidates_set & valid_set) > 0:
            candidate_models = sorted(list(candidates_set & valid_set))
            n_sel_valid = candidate_models

        elif len(candidates_set & test_set) > 0:
            candidate_models = sorted(list(candidates_set & test_set))
            n_sel_test = candidate_models

        # This has to be the case
        else:
            n_sel_test = []
            n_sel_valid = []

        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=candidate_models)

        # Save the ensemble for later use in the main auto-sklearn module!
        if ensemble is not None:
            self.backend.save_ensemble(ensemble, iteration, self.seed)

        # Delete files of non-candidate models - can only be done after fitting the
        # ensemble and saving it to disc so we do not accidentally delete models in
        # the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration, we should do this
        # before doing predictions as this is a likely place of memory issues
        with self.runs_path.open("wb") as f:
            pickle.dump(self.runs, f)

        if ensemble is not None:
            train_pred = self.predict(
                set_="train",
                ensemble=ensemble,
                selected_keys=candidate_models,
                n_preds=len(candidate_models),
                index_run=iteration,
            )
            # We can't use candidate_models here, as n_sel_* might be empty
            valid_pred = self.predict(
                set_="valid",
                ensemble=ensemble,
                selected_keys=n_sel_valid,
                n_preds=len(candidate_models),
                index_run=iteration,
            )
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(
                set_="test",
                ensemble=ensemble,
                selected_keys=n_sel_test,
                n_preds=len(candidate_models),
                index_run=iteration,
            )

            # Add a score to run history to see ensemble progress
            self._add_ensemble_trajectory(train_pred, valid_pred, test_pred)

        # The loaded predictions and hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with self.run_predictions_path.open("wb") as f:
            item = (self.run_predictions, self.last_hash)
            pickle.dump(item, f)

        if return_predictions:
            return (
                self.ensemble_history,
                self.ensemble_nbest,
                train_pred,
                valid_pred,
                test_pred,
            )
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None, None

    def run_loss(
        self,
        run: Run,
        targets: np.ndarray,
        kind: Literal["ensemble", "val", "test"] = "ensemble",
    ) -> None:
        """Compute the loss of a run on a given set of targets

        NOTE
        ----
        Still has a side effect of populating self.read_preds

        Parameters
        ----------
        run: Run
            The run to calculate the loss of

        targets: np.ndarray
            The targets for which to calculate the losses on.
            Typically the ensemble_targts.

        targets: np.ndarray
            The targets to compare against

        kind: "ensemble" | "val" | "test" = "ensemble"
            What kind of predicitons to laod from the Runs

        """
        # Put an entry in for the predictions if it doesn't exist
        if run.id not in self.run_predictions:
            self.run_predictions[run.id] = {
                Y_ENSEMBLE: None,
                Y_VALID: None,
                Y_TEST: None,
            }

        try:
            run_predictions = run.predictions("ensemble", precision=self.precision)
            loss = calculate_loss(
                solution=targets,
                prediction=run_predictions,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=None,
            )

        except Exception:
            self.logger.error(
                f"Error {kind} predictions for {run}:" f" {traceback.format_exc()}"
            )
            loss = np.inf

        finally:
            run.loss = loss
            run.loaded = 2

    def get_nbest(
        self,
        runs: Sequence[Run],
        nbest: int | None = None,
    ) -> list[Run]:
        """Get best n predictions according to the loss on the "ensemble set"

        Side effects:
        * Define the n-best models to use in ensemble
        * Only the best models are loaded
        * Any model that is not best is deletable if max models in disc is exceeded.

        Returns
        -------
        list[str]
            Returns the paths of the selected models which are used as keys in
            `run_predictions` and `runs`
        """
        if nbest is None:
            nbest = self.ensemble_nbest

        # Getting the candidates
        #
        #   First we must split out dummy runs and real runs. We sort the dummy
        #   runs to then remove any real ones that are worse than the best dummy.
        #   If this removes all viable candidates, then we reinclude dummy runs
        #   as being viable candidates.
        #
        dummies, real = itersplit(runs, func=lambda r: r.is_dummy())

        if len(dummies) == 0:
            raise ValueError("We always expect a dummy run, i.e. a run with num_run=1")

        dummy_loss = sorted(dummies)[0].loss
        self.logger.debug(f"Using {dummy_loss} to filter candidates")

        candidates = [r for r in real if r.loss < dummy_loss]

        # If there are no candidates left, use the dummies
        if len(candidates) == 0:
            if len(real) > len(dummies):
                self.logger.warning(
                    "No models better than random - using Dummy loss!"
                    f"\n\tNumber of models besides current dummy model: {len(real)}"
                    f"\n\tNumber of dummy models: {len(dummies)}",
                )

            candidates = [d for d in dummies if d.seed == self.seed]

        # Sort the candidates by lowest loss first and then lowest numrun going forward
        candidates = sorted(candidates, key=lambda r: (r.loss, r.num_run))

        # Calculate `keep_nbest` to determine how many models to keep
        #
        #   1. First we use the parameter `ensemble_nbest` to determine a base
        #   size of how many to keep, `int` being absolute and float being
        #   percentage of the available candidates.
        #
        #   2. If `max_models_on_disc` was an int, we can take this to be absolute.
        #   Otherwise, we take it to be a memory *cutoff*. We also add some buffer
        #   to the *cutoff*, essentially giving us that the *cutoff* is
        #
        #       cutoff = max_models_on_disc - size_of_largest_model
        #
        #       We use the fact models are sorted based on loss, from best to worst,
        #   and we calculate the cumulative memory cost. From this, we determine
        #   how many of the best models we can keep before we go over this *cutoff*.
        #   This is called the `max_resident_models`.
        #
        #   3. Finally, we take the smaller of the two from step 1. and 2. to determine
        #   the amount of models to keep
        #
        # Use `ensemble_n_best`
        n_candidates = len(candidates)
        if isinstance(self.ensemble_nbest, int):
            keep_nbest = min(self.ensemble_nbest, n_candidates)
        else:
            val = n_candidates * self.ensemble_nbest
            keep_nbest = int(bound(val, low=1, high=n_candidates))

        percent = keep_nbest / n_candidates
        self.logger.debug(f"Using top {keep_nbest} of {n_candidates} ({percent:.2%})")

        # Determine `max_resident_models`
        self.max_resident_models = self.max_models_on_disc
        if isinstance(self.max_resident_models, float):
            largest_mem = max(candidates, key=lambda r: r.mem_usage)
            cutoff = self.max_models_on_disc - largest_mem

            total = sum(r.mem_usage for r in candidates)
            if total <= cutoff:
                self.max_resident_models = None
            else:
                # Index of how many models before we go over the cutoff
                mem_usage_for_n_models = accumulate(r.mem_usage for r in candidates)
                max_models = findwhere(
                    mem_usage_for_n_models,
                    lambda cost: cost > cutoff,
                    default=len(candidates),
                )

                # Ensure we always at least have 1, even if the very first
                # model would have put us over the cutoff
                self.max_resident_models = max(1, max_models)

                self.logger.warning(
                    f"Limiting num of models via `max_models_on_disc` float"
                    f" max_models_on_disc={self.max_models_on_disc}"
                    f" cutoff={cutoff}"
                    f" worst={largest_mem}"
                    f" num_models={self.max_resident_models}"
                )

        if (
            self.max_resident_models is not None
            and self.max_resident_models < keep_nbest
        ):
            self.logger.debug(
                f"Restricting the number of models to {self.max_resident_models}"
                f"instead of {keep_nbest} due to argument "
            )
            keep_nbest = self.max_resident_models

        # consider performance_range_threshold
        #
        #
        if self.performance_range_threshold > 0:
            best = runs[0].loss
            cutoff = dummy_loss - (dummy_loss - best) * self.performance_range_threshold

            considered = candidates[:keep_nbest]
            if considered[-1].loss > cutoff:
                # Find the first run that is worse than the cutoff
                cutoff_run_idx = findwhere(
                    considered,
                    lambda r: r.loss >= cutoff,
                    default=len(considered),
                )

                # Make sure we always keep at least 1
                keep_nbest = max(1, cutoff_run_idx)

        keep, unload = candidates[:keep_nbest], candidates[keep_nbest:]

        # remove loaded predictions for non-winning models
        for run in unload:
            if run.id in self.run_predictions:
                self.run_predictions[run.id][Y_ENSEMBLE] = None
                self.run_predictions[run.id][Y_VALID] = None
                self.run_predictions[run.id][Y_TEST] = None

            if run.loaded == 1:
                self.logger.debug(f"Dropping model {run}")
                run.loaded = 2

        # Load the predictions for the winning
        for run in keep:
            if run.loaded != 3 and (
                run.id not in self.run_predictions
                or self.run_predictions[run.id][Y_ENSEMBLE] is None
            ):
                # No need to load valid and test here because they are loaded only if
                # the model ends up in the ensemble
                predictions = run.predictions("ensemble", precision=self.precision)
                self.run_predictions[run.id][Y_ENSEMBLE] = predictions
                run.loaded = 1

        return keep

    def get_valid_test_preds(
        self,
        selected_keys: list[str],
    ) -> tuple[list[str], list[str]]:
        """Get valid and test predictions from disc and store in self.run_predictions

        Parameters
        ----------
        selected_keys: list
            list of selected keys of self.run_predictions

        Return
        ------
        keys_valid: list[str], keys_test: list[str]
            All keys in selected keys for which we could read the valid and test
            predictions.
        """
        success_keys_valid = []
        success_keys_test = []

        for k in selected_keys:
            run = self.runs[k]

            rundir = Path(self.backend.get_numrun_directory(*run.id))

            valid_fn = rundir / f"predictions_valid_{run}.npy"
            test_fn = rundir / f"predictions_test_{run}.npy"

            if valid_fn.exists():
                if (
                    run.mtime_valid == valid_fn.stat().st_mtime
                    and k in self.run_predictions
                    and self.run_predictions[k][Y_VALID] is not None
                ):
                    success_keys_valid.append(k)
                    continue

                else:
                    try:
                        y_valid = self._predictions_from(valid_fn)
                        self.run_predictions[k][Y_VALID] = y_valid
                        success_keys_valid.append(k)
                        run.mtime_valid = valid_fn.stat().st_mtime

                    except Exception:
                        self.logger.warning(f"Err {valid_fn}:{traceback.format_exc()}")

            if test_fn.exists():
                if (
                    run.mtime_test == test_fn.stat().st_mtime
                    and k in self.run_predictions
                    and self.run_predictions[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)

                else:
                    try:
                        y_test = self._predictions_from(test_fn)
                        self.run_predictions[k][Y_TEST] = y_test
                        success_keys_test.append(k)
                        run.mtime_test = os.path.getmtime(test_fn)
                    except Exception:
                        self.logger.warning(f"Err {test_fn}:{traceback.format_exc()}")

        return success_keys_valid, success_keys_test

    def fit_ensemble(self, selected_keys: list[str]) -> EnsembleSelection:
        """TODO

        Parameters
        ----------
        selected_keys: list[str]
            List of selected keys of self.runs

        Returns
        -------
        ensemble: EnsembleSelection
            The trained ensemble
        """
        predictions_train = [self.run_predictions[k][Y_ENSEMBLE] for k in selected_keys]

        selected_runs = [self.runs[k] for k in selected_keys]

        # List of (seed, num_run, budget)
        include_num_runs = [run.id for run in selected_runs]

        # check hash if ensemble training data changed
        # TODO could we just use the size, and the last row?
        current_hash = "".join(
            [
                str(zlib.adler32(predictions_train[i].data.tobytes()))
                for i in range(len(predictions_train))
            ]
        )
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                f"-- current performance: {self.validation_performance_}",
            )
            return None

        self._last_hash = current_hash

        ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            task_type=self.task_type,
            metric=self.metric,
            random_state=self.random_state,
        )

        try:
            self.logger.debug(f"Fitting ensemble on {len(predictions_train)} models")

            start_time = time.time()

            # TODO y_ensemble can be None here
            ensemble.fit(predictions_train, self.y_ensemble, include_num_runs)

            duration = time.time() - start_time

            self.logger.debug(f"Fitting the ensemble took {duration} seconds.")
            self.logger.info(ensemble)

            ens_perf = ensemble.get_validation_performance()
            self.validation_performance_ = min(self.validation_performance_, ens_perf)

        except Exception as e:
            self.logger.error(f"Caught error {e}: {traceback.format_exc()}")
            ensemble = None
        finally:
            # Explicitly free memory
            del predictions_train
            return ensemble

    def predict(
        self,
        set_: str,
        ensemble: AbstractEnsemble,
        selected_keys: list,
        n_preds: int,
        index_run: int,
    ) -> np.ndarray | None:
        """Save preditions on ensemble, validation and test data on disc

        Parameters
        ----------
        set_: "valid" | "test" | str
            The data split name, returns preds for y_ensemble if not "valid" or "test"

        ensemble: EnsembleSelection
            The trained Ensemble

        selected_keys: list[str]
            List of selected keys of self.runs

        n_preds: int
            Number of prediction models used for ensemble building same number of
            predictions on valid and test are necessary

        index_run: int
            n-th time that ensemble predictions are written to disc

        Return
        ------
        np.ndarray | None
            Returns the predictions if it can, else None
        """
        self.logger.debug("Predicting the %s set with the ensemble!", set_)

        if set_ == "valid":
            pred_set = Y_VALID
        elif set_ == "test":
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE
        predictions = [self.run_predictions[k][pred_set] for k in selected_keys]

        if n_preds == len(predictions):
            y = ensemble.predict(predictions)
            if self.task_type == BINARY_CLASSIFICATION:
                y = y[:, 1]
            return y
        else:
            self.logger.info(
                "Found inconsistent number of predictions and models (%d vs "
                "%d) for subset %s",
                len(predictions),
                n_preds,
                set_,
            )
            return None

    def _add_ensemble_trajectory(
        self,
        train_pred: np.ndarray,
        valid_pred: np.ndarray | None,
        test_pred: np.ndarray | None,
    ) -> None:
        """
        Records a snapshot of how the performance look at a given training
        time.

        Parameters
        ----------
        train_pred: np.ndarray
            The training predictions

        valid_pred: np.ndarray | None
            The predictions on the validation set using ensemble

        test_pred: np.ndarray | None
            The predictions on the test set using ensemble
        """
        if self.task_type == BINARY_CLASSIFICATION:
            if len(train_pred.shape) == 1 or train_pred.shape[1] == 1:
                train_pred = np.vstack(
                    ((1 - train_pred).reshape((1, -1)), train_pred.reshape((1, -1)))
                ).transpose()

            if valid_pred is not None and (
                len(valid_pred.shape) == 1 or valid_pred.shape[1] == 1
            ):
                valid_pred = np.vstack(
                    ((1 - valid_pred).reshape((1, -1)), valid_pred.reshape((1, -1)))
                ).transpose()

            if test_pred is not None and (
                len(test_pred.shape) == 1 or test_pred.shape[1] == 1
            ):
                test_pred = np.vstack(
                    ((1 - test_pred).reshape((1, -1)), test_pred.reshape((1, -1)))
                ).transpose()

        # TODO y_ensemble can be None here
        performance_stamp = {
            "Timestamp": pd.Timestamp.now(),
            "ensemble_optimization_score": calculate_score(
                solution=self.y_ensemble,
                prediction=train_pred,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=None,
            ),
        }
        if valid_pred is not None:
            # TODO: valid_pred are a legacy from competition manager
            # and this if never happens. Re-evaluate Y_valid support
            performance_stamp["ensemble_val_score"] = calculate_score(
                solution=self.y_valid,
                prediction=valid_pred,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=None,
            )

        # In case test_pred was provided
        if test_pred is not None:
            performance_stamp["ensemble_test_score"] = calculate_score(
                solution=self.y_test,
                prediction=test_pred,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=None,
            )

        self.ensemble_history.append(performance_stamp)

    def _delete_excess_models(self, selected_keys: list[str]) -> None:
        """
        Deletes models excess models on disc. self.max_models_on_disc
        defines the upper limit on how many models to keep.
        Any additional model with a worst loss than the top
        self.max_models_on_disc is deleted.

        Parameters
        ----------
        selected_keys: list[str]
            TODO
        """
        # Loop through the files currently in the directory
        for pred_path in self.run_ensemble_prediction_paths:

            # Do not delete candidates
            if pred_path in selected_keys:
                continue

            match = self.model_fn_re.search(pred_path)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            # Do not delete the dummy prediction
            if _num_run == 1:
                continue

            numrun_dir = self.backend.get_numrun_directory(_seed, _num_run, _budget)
            try:
                os.rename(numrun_dir, numrun_dir + ".old")
                shutil.rmtree(numrun_dir + ".old")

                self.logger.info(f"Deleted files of non-candidate model {pred_path}")

                self.runs[pred_path].disc_space_cost_mb = None
                self.runs[pred_path].loaded = 3
                self.runs[pred_path].loss = np.inf

            except Exception as e:
                self.logger.error(
                    f"Failed to delete files of non-candidate model {pred_path} due"
                    f" to error {e}",
                )
