from __future__ import annotations

from typing import Any, Tuple

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
from pathlib import Path

import numpy as np
import pandas as pd
import pynisher

from autosklearn.automl_common.common.ensemble_building.abstract_ensemble import (  # noqa: E501
    AbstractEnsemble,
)
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import Scorer, calculate_loss, calculate_score
from autosklearn.util.disk import sizeof
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules
from autosklearn.util.functional import intersection

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2

MODEL_FN_RE = r"_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)\.npy"


class EnsembleBuilder:

    model_fn_re = re.compile(MODEL_FN_RE)

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

        # The cached values of the true targets for the ensemble
        self.y_true_ensemble: int | None = None

        # Track the ensemble performance
        self.ensemble_history = []

        # Setup the logger
        self.logger = get_named_client_logger(name="EnsembleBuilder", port=logger_port)
        self.logger_port = logger_port

        # Keep running knowledge of its validation performance
        self.validation_performance_ = np.inf

        # Data we may need
        datamanager = self.backend.load_datamanager()
        self.y_valid = datamanager.data.get("Y_valid")
        self.y_test = datamanager.data.get("Y_test")

        # Log the behaviour
        if ensemble_nbest == 1:
            t = type(ensemble_nbest)
            self.logger.debug(f"Using behaviour when {t} for {ensemble_nbest}:{t}")

        # The cached set of run_predictions which could come from previous instances
        self._run_predictions: dict[str, dict[int, np.ndarray]] | None = None

        # Hash of the last ensemble training data to identify it
        self._last_hash: str | None = None

        # The cached info of runs which could come from previous instances
        self._run_info: dict[str, dict[str, Any]] | None = None

    @property
    def run_predictions_path(self) -> Path:
        """Path to the cached predictions we store between runs"""
        return Path(self.backend.internals_directory) / "ensemble_read_preds.pkl"

    @property
    def run_info_path(self) -> Path:
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
    def run_info(self) -> dict[str, dict[str, Any]]:
        """Get the cached information from previous runs
        {
            "file name": {
                "ens_loss": float
                "mtime_ens": str,
                "mtime_valid": str,
                "mtime_test": str,
                "seed": int,
                "num_run": int,
            }
        }
        """
        if self._run_info is None:
            self._run_info = {}

            path = self.run_info_path
            if path.exists():
                with path.open("rb") as memory:
                    self._run_info = pickle.load(memory)

        return self._run_info

    def run(
        self,
        iteration: int,
        pynisher_context: str,
        time_left: float | None = None,
        end_at: float | None = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
    ) -> Tuple[
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

        pynisher_context : str
            The pynisher context to run in

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
    ) -> Tuple[
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
        self.logger.debug(
            "Starting iteration %d, time left: %f",
            iteration,
            time_left - used_time,
        )

        # populates self.run_predictions and self.run_info
        if not self.compute_loss_per_model():
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

        # If valid/test predictions loaded, then reduce candidate models to this set
        # If any of n_sel_* is not empty and overlaps with candidate_models,
        # then ensure candidate_models AND n_sel_test are sorted the same
        candidates_set = set(candidate_models)
        valid_set = set(n_sel_valid)
        test_set = set(n_sel_test)

        intersect = intersection(candidates_set, valid_set, test_set)
        if len(intersect) > 0:
            candidate_models = sorted(list(intersect))
            n_sel_test = candidate_models
            n_sel_valid = candidate_models

        elif len(candidates_set & valid_set) > 0:
            candidate_models = sorted(list(candidates_set & valid_set))
            n_sel_valid = candidate_models

        elif len(candidates_set & n_sel_test) > 0:
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
        with self.run_info_path.open("wb") as f:
            pickle.dump(self.run_info, f)

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

    def compute_loss_per_model(self) -> bool:
        """Compute the loss of the predictions on ensemble building data set;
        populates self.run_predictions and self.run_info

        Side-effects
        ------------
        * Populates
            - `self.y_ens_files` all the ensemble predictions it could find for runs
            - `self.run_info` with the new losses it calculated

        Returns
        -------
        bool
            Whether it successfully computed losses
        """
        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

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
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug(
                "Found no prediction files on ensemble data set:" " %s" % pred_path
            )
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            mtime = os.path.getmtime(y_ens_fn)

            to_read.append([y_ens_fn, match, _seed, _num_run, _budget, mtime])

        n_read_files = 0
        # Now read file wrt to num_run
        for y_ens_fn, match, _seed, _num_run, _budget, mtime in sorted(
            to_read, key=lambda x: x[5]
        ):
            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info(
                    "Error loading file (not .npy or .npy.gz): %s", y_ens_fn
                )
                continue

            if not self.run_info.get(y_ens_fn):
                self.run_info[y_ens_fn] = {
                    "ens_loss": np.inf,
                    "mtime_ens": 0,
                    "mtime_valid": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0,
                }
            if not self.run_predictions.get(y_ens_fn):
                self.run_predictions[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_VALID: None,
                    Y_TEST: None,
                }

            if self.run_info[y_ens_fn]["mtime_ens"] == mtime:
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._predictions_from(y_ens_fn)
                loss = calculate_loss(
                    solution=self.y_true_ensemble,
                    prediction=y_ensemble,
                    task_type=self.task_type,
                    metric=self.metric,
                    scoring_functions=None,
                )

                if np.isfinite(self.run_info[y_ens_fn]["ens_loss"]):
                    self.logger.debug(
                        "Changing ensemble loss for file %s from %f to %f "
                        "because file modification time changed? %f - %f",
                        y_ens_fn,
                        self.run_info[y_ens_fn]["ens_loss"],
                        loss,
                        self.run_info[y_ens_fn]["mtime_ens"],
                        os.path.getmtime(y_ens_fn),
                    )

                self.run_info[y_ens_fn]["ens_loss"] = loss

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                self.run_info[y_ens_fn]["mtime_ens"] = os.path.getmtime(y_ens_fn)
                self.run_info[y_ens_fn]["loaded"] = 2
                mem_usage = round(sizeof(y_ens_fn, unit="MB"), 2)
                self.run_info[y_ens_fn]["disc_space_cost_mb"] = mem_usage

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    "Error loading %s: %s",
                    y_ens_fn,
                    traceback.format_exc(),
                )
                self.run_info[y_ens_fn]["ens_loss"] = np.inf

        self.logger.debug(
            "Done reading %d new prediction files. Loaded %d predictions in " "total.",
            n_read_files,
            np.sum([pred["loaded"] > 0 for pred in self.run_info.values()]),
        )
        return True

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
            `run_predictions` and `run_info`
        """
        # Sort by loss - smaller is better!
        sorted_keys = list(
            sorted(
                [(k, v["ens_loss"], v["num_run"]) for k, v in self.run_info.items()],
                # Sort by loss as priority 1 and then by num_run on a ascending order
                # We want small num_run first
                key=lambda x: (x[1], x[2]),
            )
        )

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
                (k, v["ens_loss"], v["num_run"])
                for k, v in self.run_info.items()
                if v["seed"] == self.seed and v["num_run"] == 1
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
            self.logger.debug(
                "Library Pruning: using for ensemble only "
                " %d (out of %d) models" % (keep_nbest, len(sorted_keys))
            )

        # If max_models_on_disc is None, do nothing
        # One can only read at most max_models_on_disc models
        if self.max_models_on_disc is not None:
            if not isinstance(self.max_models_on_disc, numbers.Integral):
                consumption = [
                    [
                        v["ens_loss"],
                        v["disc_space_cost_mb"],
                    ]
                    for v in self.run_info.values()
                    if v["disc_space_cost_mb"] is not None
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
            if self.run_info[k]["loaded"] == 1:
                self.logger.debug(
                    "Dropping model %s (%d,%d) with loss %f.",
                    k,
                    self.run_info[k]["seed"],
                    self.run_info[k]["num_run"],
                    self.run_info[k]["ens_loss"],
                )
                self.run_info[k]["loaded"] = 2

        # Load the predictions for the winning
        for k in sorted_keys[:ensemble_n_best]:
            if (
                k not in self.run_predictions
                or self.run_predictions[k][Y_ENSEMBLE] is None
            ) and self.run_info[k]["loaded"] != 3:
                self.run_predictions[k][Y_ENSEMBLE] = self._predictions_from(k)
                # No need to load valid and test here because they are loaded
                #  only if the model ends up in the ensemble
                self.run_info[k]["loaded"] = 1

        # return keys of self.run_info with lowest losses
        return sorted_keys[:ensemble_n_best]

    def get_valid_test_preds(
        self,
        selected_keys: list[str],
    ) -> Tuple[list[str], list[str]]:
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
            info = self.run_info[k]
            seed, num_run, budget = (info["seed"], info["num_run"], info["budget"])

            rundir = Path(self.backend.get_numrun_directory(seed, num_run, budget))
            valid_fn = rundir / f"predictions_valid_{seed}_{num_run}_{budget}.npy"
            test_fn = rundir / f"predictions_test_{seed}_{num_run}_{budget}.npy"

            if valid_fn.exists():
                if (
                    self.run_info[k]["mtime_valid"] == valid_fn.stat().st_mtime
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
                        self.run_info[k]["mtime_valid"] = valid_fn.stat().st_mtime
                    except Exception:
                        self.logger.warning(f"Err {valid_fn}:{traceback.format_exc()}")

            if test_fn.exists():
                if (
                    self.run_info[k]["mtime_test"] == test_fn.stat().st_mtime
                    and k in self.run_predictions
                    and self.run_predictions[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)
                else:

                    try:
                        y_test = self._predictions_from(test_fn)
                        self.run_predictions[k][Y_TEST] = y_test
                        success_keys_test.append(k)
                        self.run_info[k]["mtime_test"] = os.path.getmtime(test_fn)
                    except Exception:
                        self.logger.warning(f"Err {test_fn}:{traceback.format_exc()}")

        return success_keys_valid, success_keys_test

    def fit_ensemble(self, selected_keys: list[str]) -> EnsembleSelection:
        """TODO

        Parameters
        ----------
        selected_keys: list[str]
            List of selected keys of self.run_info

        Returns
        -------
        ensemble: EnsembleSelection
            The trained ensemble
        """
        predictions_train = [self.run_predictions[k][Y_ENSEMBLE] for k in selected_keys]
        include_num_runs = [
            (
                self.run_info[k]["seed"],
                self.run_info[k]["num_run"],
                self.run_info[k]["budget"],
            )
            for k in selected_keys
        ]

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
            ensemble.fit(predictions_train, self.y_true_ensemble, include_num_runs)

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
            List of selected keys of self.run_info

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

        performance_stamp = {
            "Timestamp": pd.Timestamp.now(),
            "ensemble_optimization_score": calculate_score(
                solution=self.y_true_ensemble,
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
        """
        # Loop through the files currently in the directory
        for pred_path in self.y_ens_files:

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
                self.logger.info("Deleted files of non-candidate model %s", pred_path)
                self.run_info[pred_path]["disc_space_cost_mb"] = None
                self.run_info[pred_path]["loaded"] = 3
                self.run_info[pred_path]["ens_loss"] = np.inf
            except Exception as e:
                self.logger.error(
                    "Failed to delete files of non-candidate model %s due"
                    " to error %s",
                    pred_path,
                    e,
                )

    def _predictions_from(self, path: str | Path) -> np.ndarray:
        if isinstance(path, str):
            path = Path(path)

        precision = self.precision

        with path.open("rb") as f:
            predictions = np.load(f)

        dtypes = {
            16: np.float16,
            32: np.float32,
            64: np.float64,
        }
        dtype = dtypes.get(precision, predictions.dtype)
        predictions = predictions.astype(dtype=dtype, copy=False)

        return predictions
