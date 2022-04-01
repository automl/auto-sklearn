from __future__ import annotations

from typing import Any

import glob
import logging.handlers
import multiprocessing
import numbers
import os
import pickle
import re
import shutil
import time
import traceback
import zlib
from dataclasses import dataclass
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
from autosklearn.util.functional import intersection
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2


@dataclass
class Run:
    """Dataclass for storing information about a run"""

    seed: int
    num_run: int
    ens_file: str
    dir: Path
    budget: float = 0.0
    loss: float = np.inf
    _mem_usage: int | None = None
    # The recorded time of ensemble/test/valid predictions modified
    recorded_mtime_ensemble: float = 0
    recorded_mtime_test: float = 0
    recorded_mtime_valid: float = 0
    # Lazy keys so far:
    # 0 - not loaded
    # 1 - loaded and in memory
    # 2 - loaded but dropped again
    # 3 - deleted from disk due to space constraints
    loaded: int = 0

    def is_dummy(self) -> bool:
        """Whether this run is a dummy run or not"""
        return self.num_run == 1

    def was_modified(self, kind: Literal["ensemble", "valid", "test"]) -> bool:
        """Query for when the ens file was last modified"""
        # I didn't like the idea of putting this into a dict, feel free to change
        if kind == "ensemble":
            mtime = self.recorded_mtime_ensemble
        elif kind == "valid":
            mtime = self.recorded_mtime_valid
        elif kind == "test":
            mtime == self.recorded_mtime_test
        else:
            raise NotImplementedError()

        if mtime == 0:
            raise ValueError(f"Run has no recorded time for {kind}: {self}")

        return self.pred_path(kind).stat().st_mtime == mtime

    def pred_path(self, kind: Literal["ensemble", "valid", "test"]) -> Path:
        """Get the path to certain predictions"""
        fname = f"predictions_{kind}_{self.seed}_{self.num_run}_{self.budget}.npy"
        return self.dir / fname

    @property
    def mem_usage(self) -> float:
        if self._mem_usage is None:
            self._mem_usage = round(sizeof(self.dir, unit="MB"), 2)

        return self._mem_usage

    @property
    def id(self) -> tuple[int, int, float]:
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
        name = path.name
        seed, num_run, budget = name.split('_')
        return Run(seed=seed, num_run=num_run, budget=budget, dir=dir)


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
        read_at_most: int = 5,
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

        read_at_most: int = 5
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

        # max_resident_models keeps the maximum number of models in disc
        self.max_resident_models: int | None = None

        # The starting time of the procedure
        self.start_time = 0

        # Track the ensemble performance
        self.ensemble_history = []

        # Keep running knowledge of its validation performance
        self.validation_performance_ = np.inf

        # Data we may need
        datamanager = self.backend.load_datamanager()
        self.y_valid: np.ndarray | None = datamanager.data.get("Y_valid", None)
        self.y_test: np.ndarray | None = datamanager.data.get("Y_test", None)
        self._y_ensemble: np.ndarray | None = None

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
    def runs(self) -> dict[str, Run]:
        """Get the cached information from previous runs"""
        if self._runs is None:
            self._runs = {}

            # First read in all the runs on disk
            rundir = Path(self.backend.get_runs_directory())
            runs_dirs = list(rundir.iterdir())
            pred_path = os.path.join(
                glob.escape(self.backend.get_runs_directory()),
                "%d_*_*" % self.seed,
                "predictions_ensemble_%s_*_*.npy*" % self.seed,
            )
            y_ens_files = glob.glob(pred_path)
            y_ens_files = [
                y_ens_file
                for y_ens_file in y_ens_files
                if y_ens_file.endswith(".npy") or y_ens_file.endswith(".npy.gz")
            ]
            self._run_prediction_paths = y_ens_files

        return self._run_prediction_paths


            # Next, get the info about runs from last read
            if self.runs_path.exists():
                with self.runs_path.open("rb") as memory:
                    previous_info = pickle.load(memory)

        return self._runs

    @property
    def y_ensemble(self) -> np.ndarray | None:
        """The ensemble targets used for training the ensemble

        It will attempt to load and cache them in memory but
        return None if it can't.

        Returns
        -------
        np.ndarray | None
            The ensemble targets, if they can be loaded
        """
        if self._y_ensemble is None:
            if os.path.exists(self.backend._get_targets_ensemble_filename()):
                self._y_ensemble = self.backend.load_targets_ensemble()

        return self._y_ensemble

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
        # Pynisher jobs inside dask 'forget'
        # the logger configuration. So we have to set it up
        # accordingly
        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, valid_pred, test_pred = None, None, None

        used_time = time.time() - self.start_time
        left_for_iter = time_left - used_time
        self.logger.debug(f"Starting iteration {iteration}, time left: {left_for_iter}")

        # No predictions found, exit early
        if len(self.run_ensemble_prediction_paths) == 0:
            self.logger.debug("Found no predictions on ensemble data set")
            return self.ensemble_history, self.ensemble_nbest, None, None, None

        # Can't load data, exit early
        if not os.path.exists(self.backend._get_targets_ensemble_filename()):
            self.logger.debug(f"No targets for ensemble: {traceback.format_exc()}")
            return self.ensemble_history, self.ensemble_nbest, None, None, None

        self.compute_loss_per_model(targets=self.y_ensemble)

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_n_best_preds()
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

    def compute_loss_per_model(self, targets: np.ndarray) -> None:
        """Compute the loss of the predictions on ensemble building data set;
        populates self.run_predictions and self.runs

        Side-effects
        ------------
        * Populates
            - `self.runs` with the new losses it calculated

        Parameters
        ----------
        targets: np.ndarray
            The targets for which to calculate the losses on.
            Typically the ensemble_targts.
        """
        self.logger.debug("Read ensemble data set predictions")

        # First sort files chronologically
        to_read = []
        for pred_path in self.run_ensemble_prediction_paths:
            match = self.model_fn_re.search(pred_path)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            mtime = os.path.getmtime(pred_path)

            to_read.append([pred_path, match, _seed, _num_run, _budget, mtime])

        n_read_files = 0
        # Now read file wrt to num_run
        for pred_path, match, _seed, _num_run, _budget, mtime in sorted(
            to_read, key=lambda x: x[5]
        ):

            # Break out if we've read more files than we should
            if self.read_at_most is not None and n_read_files >= self.read_at_most:
                break

            if not pred_path.endswith(".npy"):
                self.logger.warning(f"Error loading file (not .npy): {pred_path}")
                continue

            # Get the run, creating one if it doesn't exist
            if pred_path not in self.runs:
                run = Run(
                    seed=_seed,
                    num_run=_num_run,
                    budget=_budget,
                    ens_file=pred_path,
                )
                self.runs[pred_path] = run
            else:
                run = self.runs[pred_path]

            # Put an entry in for the predictions if it doesn't exist
            if pred_path not in self.run_predictions:
                self.run_predictions[pred_path] = {
                    Y_ENSEMBLE: None,
                    Y_VALID: None,
                    Y_TEST: None,
                }

            # If the timestamp is the same, nothing's changed so we can move on
            if run.mtime_ens == mtime:
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._predictions_from(pred_path)
                loss = calculate_loss(
                    solution=targets,
                    prediction=y_ensemble,
                    task_type=self.task_type,
                    metric=self.metric,
                    scoring_functions=None,
                )

                if np.isfinite(run.loss):
                    self.logger.debug(
                        f"Changing ensemble loss for file {pred_path} from {run.loss}"
                        f" to {loss} because file modification time changed?"
                        f"{run.mtime_ens} -> {run.last_modified()}"
                    )

                run.loss = loss

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                run.mtime_ens = os.path.getmtime(pred_path)
                run.loaded = 2
                run.mem_usage = run.mem_usage()

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    f"Err loading {pred_path}: {traceback.format_exc()}"
                )
                run.loss = np.inf

        n_files_read = sum([run.loaded > 0 for run in self.runs.values()])
        self.logger.debug(
            f"Done reading {n_read_files} new prediction files."
            f"Loaded {n_files_read} predictions in total."
        )

    def get_n_best_preds(self) -> list[str]:
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
        # Sort by loss as priority 1 and then by num_run on a ascending order
        # We want small id first
        keys = [(path, run.loss, run.num_run) for path, run in self.runs.items()]
        sorted_keys = sorted(keys, key=lambda x: (x[1], x[2]))

        # number of models available
        num_keys = len(sorted_keys)
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is no run_id=0)
        dummy_losses = list(filter(lambda x: x[2] == 1, sorted_keys))

        # number of dummy models
        num_dummy = len(dummy_losses)
        dummy_loss = dummy_losses[0]
        self.logger.debug("Use %f as dummy loss" % dummy_loss[1])

        # sorted_keys looks like: (k, v["ens_loss"], v["num_run"])
        # On position 1 we have the loss of a minimization problem.
        # keep only the predictions with a loss smaller than the dummy
        # prediction
        sorted_keys = filter(lambda x: x[1] < dummy_loss[1], sorted_keys)

        # remove Dummy Classifier
        sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
        if not sorted_keys:
            # no model left; try to use dummy loss (num_run==0)
            # log warning when there are other models but not better than dummy model
            if num_keys > num_dummy:
                self.logger.warning(
                    "No models better than random - using Dummy loss!"
                    "Number of models besides current dummy model: %d. "
                    "Number of dummy models: %d",
                    num_keys - 1,
                    num_dummy,
                )
            sorted_keys = [
                (path, run.loss, run.num_run)
                for path, run in self.runs.items()
                if run.seed == self.seed and run.is_dummy()
            ]
        # reload predictions if losses changed over time and a model is
        # considered to be in the top models again!
        if not isinstance(self.ensemble_nbest, numbers.Integral):
            # Transform to number of models to keep. Keep at least one
            keep_nbest = max(
                1, min(len(sorted_keys), int(len(sorted_keys) * self.ensemble_nbest))
            )
            self.logger.debug(
                "Library pruning: using only top %f percent of the models for ensemble "
                "(%d out of %d)",
                self.ensemble_nbest * 100,
                keep_nbest,
                len(sorted_keys),
            )
        else:
            # Keep only at most ensemble_nbest
            keep_nbest = min(self.ensemble_nbest, len(sorted_keys))
            self.logger.debug(f"Using {keep_nbest} of total {len(sorted_keys)} models")

        # If max_models_on_disc is None, do nothing
        # One can only read at most max_models_on_disc models
        if self.max_models_on_disc is not None:
            if not isinstance(self.max_models_on_disc, numbers.Integral):
                consumption = [
                    (
                        run.loss,
                        run.mem_usage,
                    )
                    for run in self.runs.values()
                    if run.mem_usage is not None
                ]
                max_consumption = max(c[1] for c in consumption)

                # We are pessimistic with the consumption limit indicated by
                # max_models_on_disc by 1 model. Such model is assumed to spend
                # max_consumption megabytes
                if (
                    sum(c[1] for c in consumption) + max_consumption
                ) > self.max_models_on_disc:

                    # just leave the best -- smaller is better!
                    # This list is in descending order, to preserve the best models
                    sorted_cum_consumption = (
                        np.cumsum([c[1] for c in list(sorted(consumption))])
                        + max_consumption
                    )
                    max_models = np.argmax(
                        sorted_cum_consumption > self.max_models_on_disc
                    )

                    # Make sure that at least 1 model survives
                    self.max_resident_models = max(1, max_models)
                    self.logger.warning(
                        "Limiting num of models via float max_models_on_disc={}"
                        " as accumulated={} worst={} num_models={}".format(
                            self.max_models_on_disc,
                            (sum(c[1] for c in consumption) + max_consumption),
                            max_consumption,
                            self.max_resident_models,
                        )
                    )
                else:
                    self.max_resident_models = None
            else:
                self.max_resident_models = self.max_models_on_disc

        if (
            self.max_resident_models is not None
            and keep_nbest > self.max_resident_models
        ):
            self.logger.debug(
                "Restricting the number of models to %d instead of %d due to argument "
                "max_models_on_disc",
                self.max_resident_models,
                keep_nbest,
            )
            keep_nbest = self.max_resident_models

        # consider performance_range_threshold
        if self.performance_range_threshold > 0:
            best_loss = sorted_keys[0][1]
            worst_loss = dummy_loss[1]
            worst_loss -= (worst_loss - best_loss) * self.performance_range_threshold
            if sorted_keys[keep_nbest - 1][1] > worst_loss:
                # We can further reduce number of models
                # since worst model is worse than thresh
                for i in range(0, keep_nbest):
                    # Look at most at keep_nbest models,
                    # but always keep at least one model
                    current_loss = sorted_keys[i][1]
                    if current_loss >= worst_loss:
                        self.logger.debug(
                            "Dynamic Performance range: "
                            "Further reduce from %d to %d models",
                            keep_nbest,
                            max(1, i),
                        )
                        keep_nbest = max(1, i)
                        break
        ensemble_n_best = keep_nbest

        # reduce to keys
        sorted_keys = list(map(lambda x: x[0], sorted_keys))

        # remove loaded predictions for non-winning models
        for k in sorted_keys[ensemble_n_best:]:

            if k in self.run_predictions:
                self.run_predictions[k][Y_ENSEMBLE] = None
                self.run_predictions[k][Y_VALID] = None
                self.run_predictions[k][Y_TEST] = None

            run = self.runs[k]
            if run.loaded == 1:
                self.logger.debug(
                    f"Dropping model {k} {run.seed}, {run.num_run} with loss {run.loss}"
                )
                run.loaded = 2

        # Load the predictions for the winning
        for k in sorted_keys[:ensemble_n_best]:

            run = self.runs[k]
            if run.loaded != 3 and (
                k not in self.run_predictions
                or self.run_predictions[k][Y_ENSEMBLE] is None
            ):
                # No need to load valid and test here because they are loaded only if
                # the model ends up in the ensemble
                self.run_predictions[k][Y_ENSEMBLE] = self._predictions_from(k)
                run.loaded = 1

        # return keys of self.runs with lowest losses
        return sorted_keys[:ensemble_n_best]

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

    def _predictions_from(self, path: str | Path) -> np.ndarray:
        if isinstance(path, str):
            path = Path(path)

        precision = self.precision

        with path.open("rb") as f:
            # TODO: We should probably remove this requirement. I'm not sure why model
            # predictions are being saved as pickled
            predictions = np.load(f, allow_pickle=True)

        dtypes = {16: np.float16, 32: np.float32, 64: np.float64}
        dtype = dtypes.get(precision, predictions.dtype)
        predictions = predictions.astype(dtype=dtype, copy=False)

        return predictions
