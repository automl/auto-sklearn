from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import logging.handlers
import multiprocessing
import numbers
import os
import pickle
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
from autosklearn.util.disk import rmtree
from autosklearn.util.functional import cut, findwhere, split
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules

CANDIDATES_FILENAME = "previous_ensemble_building_candidates.pkl"


class EnsembleBuilder:
    """Builds ensembles out of runs that exist in the Backend

    This is used by EnsembleBuilderManager and created in a dask-client
    every time a run finishes and there is currently no EnsembleBuilder active.
    """

    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        metric: Scorer,
        ensemble_size: int = 50,
        ensemble_nbest: int | float = 50,
        max_models_on_disc: int | float | None = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: int | None = 1024,
        read_at_most: int | None = None,
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

        ensemble_size: int = 50
            maximal size of ensemble (passed to autosklearn.ensemble.ensemble_selection)

        ensemble_nbest: int | float = 50

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

        read_at_most: int | None = None
            read at most n new prediction files in each iteration. If `None`, will read
            the predictions and calculate losses for all runs that require it.


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

        if read_at_most is not None and (read_at_most < 1 or read_at_most == np.inf):
            raise ValueError("Read at most must be int greater than 1 or None")

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
        self.performance_range_threshold = performance_range_threshold

        # Decide if self.max_models_on_disc is a memory limit or model limit
        self.max_models_on_disc: int | None = None
        self.model_memory_limit: float | None = None

        if isinstance(max_models_on_disc, int):
            self.max_models_on_disc = self.max_models_on_disc
        elif isinstance(self.max_models_on_disc, float):
            self.model_memory_limit = self.max_models_on_disc

        # The starting time of the procedure
        self.start_time: float = 0.0

        # Track the ensemble performance
        self.ensemble_history: list[dict[str, Any]] = []

        # Keep running knowledge of its validation performance
        self.validation_performance_ = np.inf

        # Data we may need
        datamanager: XYDataManager = self.backend.load_datamanager()
        self._y_test: np.ndarray | None = datamanager.data.get("Y_test", None)
        self._y_ensemble: np.ndarray | None = None

    @property
    def previous_candidates_path(self) -> Path:
        """Path to the cached losses we store between runs"""
        return Path(self.backend.internals_directory) / CANDIDATES_FILENAME

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
        runs = iter(Run(path=dir) for dir in runs_dir.iterdir() if Run.valid(dir))
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
        if kind == "ensemble":
            if self._y_ensemble is None:
                if os.path.exists(self.backend._get_targets_ensemble_filename()):
                    self._y_ensemble = self.backend.load_targets_ensemble()
            return self._y_ensemble

        elif kind == "test":
            return self._y_test

        else:
            raise NotImplementedError(kind)

    def run(
        self,
        iteration: int,
        pynisher_context: str | None = None,
        time_left: float | None = None,
        end_at: float | None = None,
        time_buffer: int = 5,
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

        Returns
        -------
        (ensemble_history, nbest)
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

            safe_ensemble_script(time_left, iteration)

            status = safe_ensemble_script.exit_status
            if isinstance(status, pynisher.MemorylimitException):
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
            elif isinstance(status, pynisher.AnythingException):
                return ([], self.ensemble_nbest)
            else:
                return safe_ensemble_script.result

        return [], self.ensemble_nbest

    def main(
        self,
        time_left: float | None = None,
        iteration: int = 0,
    ) -> tuple[list[dict[str, Any]], int | float]:
        """Run the main loop of ensemble building

        The process is:
        * Load all available runs + previous candidates (if any)
        * Update the loss of those that require
        * From these runs, get a list of candidates
        * Save candidates
        * Delete models that are not candidates
        * Build an ensemble from the candidates if there are new candidates

        Parameters
        ----------
        time_left : float | None = None
            How much time is left for this run

        iteration : int = 0
            The iteration of this run

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

        if time_left is not None:
            self.start_time = time.time()
            used_time = time.time() - self.start_time
            left_for_iter = time_left - used_time
            itr = iteration if str(iteration) is not None else ""
            self.logger.debug(f"Starting iteration {itr}, time left: {left_for_iter}")

        # Can't load data, exit early
        if not os.path.exists(self.backend._get_targets_ensemble_filename()):
            self.logger.debug(f"No targets for ensemble: {traceback.format_exc()}")
            raise RuntimeError("No targets for ensemble")

        # We will delete runs once we are complete
        deletable_runs: set[Run] = set()

        # Load in information from previous candidates and also runs
        available_runs = self.available_runs()

        # Update runs with information of available previous candidates
        previous_candidates = self.previous_candidates()
        available_runs.update(previous_candidates)

        # We just need the values now, not the key value pairs {run.id: Run}
        runs = list(available_runs.values())

        if len(runs) == 0:
            self.logger.debug("Found no runs")
            raise RuntimeError("Found no runs")

        # Calculate the loss for those that require it
        requires_update = self.requires_loss_update(runs)
        if self.read_at_most is not None:
            requires_update = requires_update[: self.read_at_most]

        for run in requires_update:
            run.record_modified_times()  # So we don't count as modified next time
            run.loss = self.loss(run, kind="ensemble")

        # Get the dummy and real runs
        dummies, candidates = split(runs, by=lambda r: r.is_dummy())

        # We see if we need to delete any of the real runs before we waste compute
        # on evaluating their candidacy for ensemble building
        if any(candidates):
            candidates, to_delete = self.requires_deletion(
                candidates,
                max_models=self.max_models_on_disc,
                memory_limit=self.model_memory_limit,
            )

            # If there are no candidates left, we just keep the best one
            if not any(candidates):
                best = min(to_delete, key=lambda r: (r.loss, r.num_run))
                candidates = [best]
                to_delete.remove(best)

            if any(to_delete):
                self.logger.info(
                    f"Deleting runs {to_delete} due to"
                    f" max_models={self.max_models_on_disc} and/or"
                    f" memory_limit={self.model_memory_limit}"
                )
                deletable_runs.update(to_delete)

        # If there are any candidates, perform candidates selection
        if any(candidates):
            candidates, to_delete = self.candidate_selection(
                runs=candidates,
                dummies=dummies,
                better_than_dummy=True,
                nbest=self.ensemble_nbest,
                performance_range_threshold=self.performance_range_threshold,
            )
            if any(to_delete):
                self.logger.info(
                    f"Deleting runs {to_delete} due to"
                    f" nbest={self.ensemble_nbest} and/or"
                    f" performance_range_threshold={self.performance_range_threshold}"
                )
                deletable_runs.update(to_delete)
        else:
            candidates = dummies
            self.logger.warning("No runs were available to build an ensemble from")

        # In case we record test predictions and not every model has test predictions,
        # only use the subset of models that has predictions for both the test set and
        # the ensemble optimization set.
        candidates_set = set(candidates)
        test_subset = {r for r in candidates if r.pred_path("test").exists()}

        if len(test_subset) > 0:
            candidates = sorted(test_subset, key=lambda r: r.id)
            test_models = candidates

            to_delete = candidates_set - test_subset
            if any(to_delete):
                self.logger.info(
                    f"Deleting runs {to_delete} due to runs not"
                    ' having "test_predictions" while others do not:'
                    f"\nHave test_predictions = {test_subset}"
                    f"\nNo test_predictions = {to_delete}"
                )
                deletable_runs.update(to_delete)

        else:
            candidates = sorted(candidates_set, key=lambda r: r.id)
            test_models = []

        # Save the candidates for the next round
        with self.previous_candidates_path.open("wb") as f:
            pickle.dump({run.id: run for run in candidates}, f)

        # If there was any change from the previous run, either in terms of
        # runs or one of those runs had its loss updated, then we need to
        # fit the ensemble builder
        previous_candidate_ids = set(previous_candidates)
        current_candidate_ids = set(run.id for run in candidates)
        difference = previous_candidate_ids ^ current_candidate_ids

        was_updated_candidates = list(run in candidates for run in requires_update)

        if not any(difference) and not any(was_updated_candidates):
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

        self.logger.info(str(ensemble))
        ens_perf = ensemble.get_validation_performance()
        self.validation_performance_ = min(self.validation_performance_, ens_perf)
        self.backend.save_ensemble(
            ensemble=ensemble, idx=iteration, seed=self.seed  # type: ignore
        )

        # Continue with evaluating the ensemble after making some space
        performance_stamp = {"Timestamp": pd.Timestamp.now()}

        for kind, score_name, models in [
            ("ensemble", "optimization", candidates),
            ("test", "test", test_models),
        ]:
            if len(models) == 0:
                continue

            pred_targets = self.targets(kind)
            if pred_targets is None:
                self.logger.warning(f"No ensemble targets for {kind}")
                continue

            run_preds = [r.predictions(kind, precision=self.precision) for r in models]
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

        # Lastly, delete any runs that need to be deleted. We save this as the last step
        # so that we have an ensemble saved that is up to date. If we do not do so,
        # there could be runs deleted that are in th previous ensemble and we do not
        # manage to update the ensemble due to a crash or the process being killed
        # before it could be updated
        self.delete_runs(deletable_runs)

        return self.ensemble_history, self.ensemble_nbest

    def requires_loss_update(
        self,
        runs: Sequence[Run],
    ) -> list[Run]:
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
        for run in sorted(runs, key=lambda run: run.recorded_mtimes["ensemble"]):
            if run.loss == np.inf:
                queue.append(run)

            elif run.was_modified():
                self.logger.debug(f"{run.id} had its predictions modified?")
                queue.append(run)

        return queue

    def candidate_selection(
        self,
        runs: Sequence[Run],
        dummies: Run | list[Run],
        *,
        better_than_dummy: bool = False,
        nbest: int | float | None = None,
        performance_range_threshold: float | None = None,
    ) -> tuple[list[Run], set[Run]]:
        """Get a list of candidates from `runs`, garuanteeing at least one

        Applies a set of reductions in order of parameters to reach a set of final
        candidates.

        Expects at least one `dummies` run.

        Parameters
        ----------
        runs : Sequence[Run]
            The runs to evaluate candidates from.

        dummies: Run | Sequence[Run]
            The dummy run to base from

        better_than_dummy: bool = False
            Whether the run must be better than the best dummy run to be a candidate.
            In the case where there are no candidates left, the dummies will then be
            used.

        nbest : int | float | None
            The nbest models to select. If `int`, acts as an absolute limit.
            If `float`, acts as a percentage of available candidates.

        model_memory_limit : float | None
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
        if isinstance(dummies, Run):
            dummies = [dummies]

        assert len(dummies) > 0 and len(runs) > 0, "At least 1 real run and dummy run"

        all_discarded: set[Run] = set()

        # We filter out all runs that don't have any predictions for the ensemble
        candidates, discarded = split(
            runs, by=lambda run: run.pred_path("ensemble").exists()
        )
        all_discarded.update(discarded)

        if len(candidates) == 0:
            self.logger.debug("No runs with predictions on ensemble set, using dummies")
            return dummies, all_discarded

        for run in discarded:
            self.logger.warning(f"Have no ensemble predictions for {run}")

        # Get all the ones that have a tangible loss
        candidates, discarded = split(
            candidates,
            by=lambda r: r.loss < np.inf,
        )
        all_discarded.update(discarded)

        if len(candidates) == 0:
            self.logger.debug("No runs with a usable loss, using dummies")
            return dummies, all_discarded

        if better_than_dummy:
            dummies = sorted(dummies, key=lambda r: r.loss)
            dummy_cutoff = dummies[0].loss
            self.logger.debug(f"Using {dummy_cutoff} to filter candidates")

            candidates, discarded = split(
                candidates,
                by=lambda r: r.loss < dummy_cutoff,
            )
            all_discarded.update(discarded)

            # If there are no real candidates left, use the dummies
            if len(candidates) == 0:
                self.logger.warning(
                    "No models better than random - using Dummy loss!"
                    f"\n\tModels besides current dummy model: {len(candidates)}"
                    f"\n\tDummy models: {len(dummies)}",
                )
                return dummies, all_discarded

        # Sort the candidates so that they ordered by best loss, using num_run for tie
        candidates = sorted(candidates, key=lambda r: (r.loss, r.num_run))

        if nbest is not None:
            # Determine how many to keep, always keeping one
            if isinstance(nbest, float):
                nkeep = int(len(candidates) * nbest)
            else:
                nkeep = nbest

            candidates, discarded = cut(candidates, nkeep)
            self.logger.info(f"Discarding {len(discarded)}/{len(candidates)} runs")

            # Always preserve at least one, the best
            if len(candidates) == 0:
                candidates, discared = cut(discarded, 1)
                self.logger.warning("nbest too aggresive, using best")

            all_discarded.update(discarded)

        if performance_range_threshold is not None:
            x = performance_range_threshold
            worst = dummies[0].loss
            best = candidates[0].loss

            cutoff = x * best + (1 - x) * worst

            candidates, discarded = cut(candidates, where=lambda r: r.loss >= cutoff)

            # Always preserve at least one, the best
            if len(candidates) == 0:
                candidates, discared = cut(discarded, 1)
                self.logger.warning("No models in performance range, using best")

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
    ) -> EnsembleSelection:
        """Fit an ensemble from the provided runs.

        Note
        ----
        Expects all runs to have the "ensemble" predictions present

        Parameters
        ----------
        runs: list[Run]
            List of runs to build an ensemble from

        targets: np.ndarray
            The targets to build the ensemble with

        size: int | None = None
            The size of the ensemble to build

        task: int | None = None
            The kind of task performed

        metric: Scorer | None = None
            The metric to use when comparing run predictions to the targets

        precision: int | None = None
            The precision with which to load run predictions

        random_state: int | RandomState | None = None
            The random state to use

        Returns
        -------
        ensemble: EnsembleSelection
            The trained ensemble
        """
        task = task if task is not None else self.task_type
        size = size if size is not None else self.ensemble_size
        metric = metric if metric is not None else self.metric
        rs = random_state if random_state is not None else self.random_state

        ensemble = EnsembleSelection(
            ensemble_size=size,
            task_type=task,
            metric=metric,
            random_state=rs,
        )

        self.logger.debug(f"Fitting ensemble on {len(runs)} models")
        start_time = time.time()

        precision = precision if precision is not None else self.precision
        predictions_train = [
            run.predictions("ensemble", precision=precision) for run in runs
        ]

        ensemble.fit(
            predictions=predictions_train,
            labels=targets,
            identifiers=[run.id for run in runs],
        )

        duration = time.time() - start_time
        self.logger.debug(f"Fitting the ensemble took {duration} seconds.")
        return ensemble

    def requires_deletion(
        self,
        runs: Sequence[Run],
        *,
        max_models: int | None = None,
        memory_limit: float | None = None,
    ) -> tuple[list[Run], set[Run]]:
        """Cut a list of runs into those to keep and those to delete

        If neither params are specified, this method should do nothing.

        Parameters
        ----------
        runs : Sequence[Run]
            The runs to check

        max_models : int | None = None
            The maximum amount of models to have on disk. Leave `None` for no effect

        memory_limit : float | None = None
            The memory limit in MB, leave `None` for no effect

        Returns
        -------
        (keep: list[Run], delete: set[Run])
            The list of runs to keep and those to delete
        """
        if memory_limit is None and max_models is None:
            return list(runs), set()

        # Start with keep all runs and dummies, deleteing None
        keep = sorted(runs, key=lambda r: (r.loss, r.num_run))
        delete: set[Run] = set()

        if max_models is not None and max_models < len(runs):
            keep, to_delete = cut(keep, max_models)

            if any(to_delete):
                delete.update(to_delete)

        if memory_limit is not None:
            largest = max(runs, key=lambda r: r.mem_usage)
            cutoff = memory_limit - largest.mem_usage

            accumulated_mem_usage = accumulate(r.mem_usage for r in runs)

            cutpoint = findwhere(accumulated_mem_usage, lambda mem: mem > cutoff)
            keep, to_delete = cut(keep, cutpoint)

            if any(to_delete):
                self.logger.warning(
                    "Limiting num of models via `memory_limit`"
                    f" memory_limit={memory_limit}"
                    f" cutoff={cutoff}"
                    f" largest={largest.mem_usage}"
                    f" remaining={len(keep)}"
                    f" discarded={len(to_delete)}"
                )
                delete.update(to_delete)

        return keep, delete

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

    def delete_runs(self, runs: Iterable[Run]) -> None:
        """Delete runs

        Will not delete dummy runs

        Parameters
        ----------
        runs : Sequence[Run]
            The runs to delete
        """
        items = iter(run for run in runs if not run.is_dummy() and run.dir.exists())
        for run in items:
            try:
                rmtree(run.dir, atomic=True)
                self.logger.info(f"Deleted files for {run}")
            except Exception as e:
                self.logger.error(f"Failed to delete files for {run}: \n{e}")
