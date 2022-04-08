from __future__ import annotations

from typing import Any, Sequence, cast

import logging.handlers
import multiprocessing
import numbers
import os
import pickle
import shutil
import time
import traceback
from itertools import accumulate
from pathlib import Path

import numpy as np
import pandas as pd
import pynisher

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.ensemble_building.run import Run, RunID
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import Scorer, calculate_loss, calculate_score
from autosklearn.util.functional import bound, cut, findwhere, intersection, split
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules


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

                x = performance_range_threshold
                x * dummy

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
        self.start_time: float = 0.0

        # Track the ensemble performance
        self.ensemble_history: list[dict[str, Any]] = []

        # Keep running knowledge of its validation performance
        self.validation_performance_ = np.inf

        # Data we may need
        datamanager: XYDataManager = self.backend.load_datamanager()
        self._y_valid: np.ndarray | None = datamanager.data.get("Y_valid", None)
        self._y_test: np.ndarray | None = datamanager.data.get("Y_test", None)
        self._y_ensemble: np.ndarray | None = None

    @property
    def previous_candidates_path(self) -> Path:
        """Path to the cached losses we store between runs"""
        fname = "previous_ensemble_building_candidates.pkl"
        return Path(self.backend.internals_directory) / fname

    def previous_candidates(self) -> dict[RunID, Run]:
        """Load any previous candidates that were saved from previous runs

        Returns
        -------
        dict[RunID, Run]
            A dictionary from RunId's to the previous candidates
        """
        if self.previous_candidates_path.exists():
            with self.previous_candidates_path.open("rb") as f:
                return pickle.load(f)
        else:
            return {}

    def available_runs(self) -> dict[RunID, Run]:
        """Get a dictionary of all available runs on the filesystem

        Returns
        -------
        dict[RunID, Run]
            A dictionary from RunId's to the available runs
        """
        runs_dir = Path(self.backend.get_runs_directory())
        runs = [Run(path=dir) for dir in runs_dir.iterdir()]
        return {run.id: run for run in runs}

    def targets(self, kind: str = "ensemble") -> np.ndarray | None:
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
    ) -> tuple[list[dict[str, Any]], int | float]:
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
                assert end_at is not None
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
                    self.previous_candidates_path.unlink()
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
                    return [], self.ensemble_nbest
            else:
                return safe_ensemble_script.result

        return [], self.ensemble_nbest

    def main(
        self,
        time_left: float,
        iteration: int,
        return_predictions: bool = False,
    ) -> tuple[list[dict[str, Any]], int | float]:
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
        (ensemble_history: list[dict[str, Any]], nbest: int | float)
        """
        # Pynisher jobs inside dask 'forget' the logger configuration.
        # So we have to set it up accordingly
        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        self.start_time = time.time()

        used_time = time.time() - self.start_time
        left_for_iter = time_left - used_time
        self.logger.debug(f"Starting iteration {iteration}, time left: {left_for_iter}")

        # Can't load data, exit early
        if not os.path.exists(self.backend._get_targets_ensemble_filename()):
            self.logger.debug(f"No targets for ensemble: {traceback.format_exc()}")
            return self.ensemble_history, self.ensemble_nbest

        # Load in information from previous candidates and also runs
        available_runs = self.available_runs()

        # Update runs with information of available previous candidates
        previous_candidates = self.previous_candidates()
        available_runs.update(previous_candidates)

        # We just need the values now, not the key value pairs {run.id: Run}
        runs = list(available_runs.values())

        if len(runs) == 0:
            self.logger.debug("Found no runs")
            return self.ensemble_history, self.ensemble_nbest

        # Calculate the loss for those that require it
        requires_update = self.requires_loss_update(runs, limit=self.read_at_most)
        for run in requires_update:
            run.loss = self.loss(run, kind="ensemble")

        # Decide if self.max_models_on_disk is an
        if isinstance(self.max_models_on_disc, int):
            max_models_on_disk = self.max_models_on_disc
            memory_limit = None
        elif isinstance(self.max_models_on_disc, float):
            max_models_on_disk = None
            memory_limit = self.max_models_on_disc
        else:
            max_models_on_disk = None
            memory_limit = None

        candidates, all_discarded = self.candidates(
            runs=runs,
            better_than_dummy=True,
            nbest=self.ensemble_nbest,
            max_models_on_disk=max_models_on_disk,
            memory_limit=memory_limit,
            performance_range_threshold=self.performance_range_threshold,
        )

        if len(candidates) == 0:
            self.logger.debug("No viable candidates found for ensemble building")
            return self.ensemble_history, self.ensemble_nbest

        # Get a set representation of them as we will begin doing intersections
        # Not here that valid_set and test_set are both subsets of candidates_set
        candidates_set = set(candidates)
        valid_set = {r for r in candidates if r.pred_path("valid").exists()}
        test_set = {r for r in candidates if r.pred_path("test").exists()}

        if len(valid_set & test_set) == 0 and len(test_set) > 0 and len(valid_set) > 0:
            self.logger.error("valid_set and test_set not empty but do not overlap")
            return self.ensemble_history, self.ensemble_nbest

        # Find the intersect between the most groups and use that to fit the ensemble
        intersect = intersection(candidates_set, valid_set, test_set)
        if len(intersect) > 0:
            candidates = list(intersect)
            candidates = sorted(candidates, key=lambda r: r.id)

            valid_models = candidates
            test_models = candidates

        elif len(candidates_set & valid_set) > 0:
            intersect = candidates_set & valid_set
            candidates, discarded = split(candidates, by=lambda r: r in intersect)
            candidates = sorted(candidates, key=lambda r: r.id)

            valid_models = candidates
            test_models = []

        elif len(candidates_set & test_set) > 0:
            intersect = candidates_set & test_set
            candidates, discarded = split(candidates, by=lambda r: r in intersect)
            candidates = sorted(candidates, key=lambda r: r.id)

            valid_models = []
            test_models = candidates

        else:
            candidates = sorted(candidates, key=lambda r: r.id)
            discarded = []

            valid_models = []
            test_models = []

        all_discarded.update(discarded)

        # To save on pickle and to allow for fresh predictions, unload the cache
        # before pickling
        for run in candidates:
            run.unload_cache()

        # Save the candidates for the next round
        with self.previous_candidates_path.open("wb") as f:
            pickle.dump({run.id: run for run in candidates}, f)

        # Delete files for models which were not considered candidates
        if len(discarded) > 0:
            for run in discarded:
                if not run.is_dummy():
                    try:
                        shutil.rmtree(run.dir)
                        self.logger.info(f"Deleted files for {run}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete files for {run}: \n{e}")

        # If there was any change from the previous run, either in terms of
        # runs or one of those runs had its loss updated, then we need to
        # fit the ensemble builder
        previous_candidate_ids = set(previous_candidates.keys())
        current_candidate_ids = set(run.id for run in candidates)
        different_candidates = previous_candidate_ids ^ current_candidate_ids

        updated_candidates = iter(run in candidates for run in requires_update)

        if not any(different_candidates) or any(updated_candidates):
            self.logger.info("All ensemble candidates the same, no update required")
            return self.ensemble_history, self.ensemble_nbest

        targets = cast(np.ndarray, self.targets("ensemble"))  # Sure they exist
        ensemble = self.fit_ensemble(
            candidates,
            targets=targets,
            size=self.ensemble_size,
            task=self.task_type,
            metric=self.metric,
            precision=self.precision,
            random_state=self.random_state,
        )

        if ensemble is not None:
            self.logger.info(str(ensemble))
            ens_perf = ensemble.get_validation_performance()
            self.validation_performance_ = min(self.validation_performance_, ens_perf)
            self.backend.save_ensemble(ensemble, iteration, self.seed)  # type: ignore

        # Continue with evaluating the ensemble after making some space
        if ensemble is not None:
            performance_stamp = {"Timestamp": pd.Timestamp.now()}

            for kind, score_name, models in [
                ("ensemble", "optimization", candidates),
                ("valid", "val", valid_models),
                ("test", "test", test_models),
            ]:
                if len(models) == 0:
                    continue

                pred_targets = self.targets(kind)
                if pred_targets is None:
                    self.logger.warning(f"No ensemble targets for {kind}")
                    continue

                run_preds = [
                    r.predictions(kind, precision=self.precision) for r in models
                ]
                pred = ensemble.predict(run_preds)

                # Pretty sure this whole step is uneeded but left over and afraid
                # to touch
                if self.task_type == BINARY_CLASSIFICATION:
                    pred = pred[:, 1]

                    if pred.ndim == 1 or pred.shape[1] == 1:
                        pred = np.vstack(
                            ((1 - pred).reshape((1, -1)), pred.reshape((1, -1)))
                        ).transpose()

                score = calculate_score(
                    solution=pred_targets,
                    prediction=pred,
                    task_type=self.task_type,
                    metric=self.metric,
                    scoring_functions=None,
                )
                performance_stamp[f"ensemble_{score_name}_score"] = score
                self.ensemble_history.append(performance_stamp)

        return self.ensemble_history, self.ensemble_nbest

    def requires_loss_update(self, runs: Sequence[Run], limit: int | None) -> list[Run]:
        """

        Parameters
        ----------
        runs : Sequence[Run]
            The runs to process

        Returns
        -------
        list[Run]
            The runs that require a loss to be calculated
        """
        queue = []
        for run in runs:
            if run.loss is None or run.loss == np.inf:
                queue.append(run)

            elif run.loss is not None and run.pred_modified("ensemble"):
                self.logger.debug(f"{run.id} had its predictions modified?")
                run.record_modified_times()  # re-mark modfied times
                queue.append(run)

        if limit is not None:
            return queue[:limit]
        else:
            return queue

    def candidates(
        self,
        runs: Sequence[Run],
        *,
        better_than_dummy: bool = False,
        nbest: int | float | None = None,
        max_models_on_disk: int | None = None,
        memory_limit: float | None = None,
        performance_range_threshold: float | None = None,
    ) -> tuple[list[Run], set[Run]]:
        """Get a list of candidates from `runs`

        Applies a set of reductions in order of parameters to reach a set of final
        candidates.

        Expects at least one `dummy` run in `runs`.

        Parameters
        ----------
        runs : Sequence[Run]
            The runs to evaluate candidates from.

        better_than_dummy: bool = False
            Whether the run must be better than the best dummy run to be a candidate.
            In the case where there are no candidates left, the dummies will then be
            used.

        nbest : int | float | None
            The nbest models to select. If `int`, acts as an absolute limit.
            If `float`, acts as a percentage of available candidates.

        max_models_on_disk : int | None
            The maximum amount of models allowed on disk. If the number of candidates
            exceed this limit after previous filters applied, this will further
            reduce the candidates.

        memory_limit : float | None
            A maximum memory limit in MB for the runs to occupy. If the candidates at
            this point exceed this limit, the best n candidates that fit into this limit
            will be chosen.

        performance_range_threshold : float | None
            A number in (0, 1) to select candidates from. Expects a dummy run for worst

        Returns
        -------
        (candidates: list[Run], discarded: set[Run])
            A tuple of runs that are candidates and also those that didn't make it
        """
        all_discarded: set[Run] = set()

        # We filter out all runs that don't have any predictions for the ensemble
        has_predictions = lambda run: run.pred_path("ensemble").exists()
        candidates, discarded = split(runs, by=has_predictions)
        all_discarded.update(discarded)

        if len(candidates) == 0:
            self.logger.debug("No runs with predictions on ensemble data set")
            return candidates, all_discarded

        if len(discarded) > 0:
            self.logger.warning(f"Have no ensemble predictions for {discarded}")

        # Get all the ones that have a tangible loss
        candidates, discarded = split(candidates, by=lambda r: r.loss < np.inf)
        all_discarded.update(discarded)

        if len(candidates) == 0:
            self.logger.debug("No runs with a usable loss")
            return candidates, all_discarded

        # Further split the candidates into those that are real and dummies
        dummies, real = split(candidates, by=lambda r: r.is_dummy())
        dummies = sorted(dummies, key=lambda r: r.loss)
        dummy_cutoff = dummies[0].loss

        if len(dummies) == 0:
            self.logger.error("Expected at least one dummy run")
            raise RuntimeError("Expected at least one dummy run")

        if len(real) == 0:
            self.logger.warning("No real runs, using dummies as candidates")
            candidates = dummies
            return candidates, all_discarded

        if better_than_dummy:
            self.logger.debug(f"Using {dummy_cutoff} to filter candidates")

            candidates, discarded = split(real, by=lambda r: r.loss < dummy_cutoff)
            all_discarded.update(discarded)

            # If there are no real candidates left, use the dummies
            if len(candidates) == 0:
                candidates = dummies
                if len(real) > 0:
                    self.logger.warning(
                        "No models better than random - using Dummy loss!"
                        f"\n\tNumber of models besides current dummy model: {len(real)}"
                        f"\n\tNumber of dummy models: {len(dummies)}",
                    )

        n_candidates = len(candidates)

        # Decide how many instanceto keep
        nkeep: int | None
        if isinstance(nbest, float):
            nkeep = int(bound(n_candidates * nbest, bounds=(1, n_candidates)))
        else:
            nkeep = nbest

        if max_models_on_disk is not None:
            if nkeep is None:
                nkeep = max_models_on_disk
            elif max_models_on_disk < nkeep:
                self.logger.warning(
                    f"Limiting {n_candidates} by"
                    f"`max_models_on_disk={max_models_on_disk}`"
                    f"instead of {nkeep} (set from `nbest={nbest}`)"
                )
                nkeep = max_models_on_disk
        else:
            nkeep = nkeep

        # Sort the candidates so that they ordered by best loss, using num_run for tie
        candidates = sorted(candidates, key=lambda r: (r.loss, r.num_run))

        # If we need to specify how many to keep, keep that many
        if nkeep is not None:
            candidates, discarded = cut(candidates, nkeep)
            all_discarded.update(discarded)
            self.logger.info(f"Discarding {len(discarded)}/{n_candidates} runs")

        # Choose which ones to discard if there's a memory limit
        if memory_limit is not None:
            largest = max(candidates, key=lambda r: r.mem_usage)
            cutoff = memory_limit - largest.mem_usage

            accumulated_mem_usage = accumulate(r.mem_usage for r in candidates)
            cutpoint = findwhere(accumulated_mem_usage, lambda mem: mem >= cutoff)

            candidates, discarded = cut(candidates, cutpoint)

            self.logger.warning(
                "Limiting num of models via `memory_limit` float"
                f" memory_limit={memory_limit}"
                f" cutoff={cutoff}"
                f" largest={largest.mem_usage}"
                f" remaining={len(candidates)}"
                f" discarded={len(discarded)}"
            )
            all_discarded.update(discarded)

        if performance_range_threshold is not None:
            x = performance_range_threshold
            worst = dummies[0].loss
            best = candidates[0].loss

            cutoff = x * best + (1 - x) * worst

            candidates, discarded = cut(candidates, where=lambda r: r.loss >= cutoff)

            all_discarded.update(discarded)

        return candidates, all_discarded

    def fit_ensemble(
        self,
        runs: list[Run],
        targets: np.ndarray,
        *,
        size: int | None = None,
        task: int | None = None,
        metric: Scorer | None = None,
        precision: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> EnsembleSelection | None:
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
        task = task if task is not None else self.task_type
        size = size if size is not None else self.ensemble_size
        metric = metric if metric is not None else self.metric
        rs = random_state if random_state is not None else self.random_state

        ensemble: EnsembleSelection | None

        ensemble = EnsembleSelection(
            ensemble_size=size,
            task_type=task,
            metric=metric,
            random_state=rs,
        )

        self.logger.debug(f"Fitting ensemble on {len(runs)} models")
        start_time = time.time()

        try:
            precision = precision if precision is not None else self.precision
            predictions_train = [
                run.predictions("ensemble", precision=precision) for run in runs
            ]

            ensemble.fit(
                predictions=predictions_train,
                labels=targets,
                identifiers=[run.id for run in runs],
            )
        except Exception as e:
            self.logger.error(f"Caught error {e}: {traceback.format_exc()}")
            ensemble = None
        finally:
            duration = time.time() - start_time
            self.logger.debug(f"Fitting the ensemble took {duration} seconds.")
            return ensemble

    def loss(self, run: Run, kind: str = "ensemble") -> float:
        """Calculate the loss for a list of runs

        Parameters
        ----------
        run: Run
            The run to calculate the loss for

        Returns
        -------
        float
            The loss for the run
        """
        targets = self.targets(kind)
        if targets is None:
            self.logger.error(f"No targets of {kind}")
            return np.inf

        try:
            predictions = run.predictions(kind, precision=self.precision)
            loss: float = calculate_loss(  # type: ignore
                solution=targets,
                prediction=predictions,
                task_type=self.task_type,
                metric=self.metric,
            )
        except Exception as e:
            self.logger.error(f"Error getting loss {run}:{e}{traceback.format_exc()}")
            loss = np.inf
        finally:
            return loss
