from __future__ import annotations

from typing import Any, Optional, Union

import logging.handlers
import time
import traceback

import dask.distributed
import numpy as np
from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import StatusType

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.ensemble_building.builder import EnsembleBuilder
from autosklearn.metrics import Scorer
from autosklearn.util.logging_ import get_named_client_logger


class EnsembleBuilderManager(IncorporateRunResultCallback):
    def __init__(
        self,
        start_time: float,
        time_left_for_ensembles: float,
        backend: Backend,
        dataset_name: str,
        task: int,
        metric: Scorer,
        ensemble_size: int,
        ensemble_nbest: int,
        seed: int,
        precision: int,
        max_iterations: Optional[int],
        read_at_most: int,
        ensemble_memory_limit: Optional[int],
        random_state: Union[int, np.random.RandomState],
        max_models_on_disc: Optional[float | int] = 100,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        pynisher_context: str = "fork",
    ):
        """SMAC callback to handle ensemble building

        Parameters
        ----------
        start_time: int
            the time when this job was started, to account for any latency in job
            allocation.

        time_left_for_ensemble: int
            How much time is left for the task. Job should finish within this
            allocated time

        backend: util.backend.Backend
            backend to write and read files

        dataset_name: str
            name of dataset

        task_type: int
            type of ML task

        metric: str
            name of metric to compute the loss of the given predictions

        ensemble_size: int
            maximal size of ensemble

        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models

        max_models_on_disc: Optional[int | float] = 100
           Defines the maximum number of models that are kept in the disc.

           If int, it must be greater or equal than 1, and dictates the max
           number of models to keep.

           If float, it will be interpreted as the max megabytes allowed of
           disc space. That is, if the number of ensemble candidates require more
           disc space than this float value, the worst models will be deleted to
           keep within this budget. Models and predictions of the worst-performing
           models will be deleted then.

           If None, the feature is disabled. It defines an upper bound on the
           models that can be used in the ensemble.

        seed: int
            random seed

        max_iterations: int
            maximal number of iterations to run this script
            (default None --> deactivated)

        precision: [16,32,64,128]
            precision of floats to read the predictions

        ensemble_memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.

        read_at_most: int
            read at most n new prediction files in each iteration

        logger_port: int
            port that receives logging records

        pynisher_context: str
            The multiprocessing context for pynisher. One of spawn/fork/forkserver.

        """
        self.start_time = start_time
        self.time_left_for_ensembles = time_left_for_ensembles
        self.backend = backend
        self.dataset_name = dataset_name
        self.task = task
        self.metric = metric
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.seed = seed
        self.precision = precision
        self.max_iterations = max_iterations
        self.read_at_most = read_at_most
        self.ensemble_memory_limit = ensemble_memory_limit
        self.random_state = random_state
        self.logger_port = logger_port
        self.pynisher_context = pynisher_context

        # Store something similar to SMAC's runhistory
        self.history: list[dict[str, Any]] = []

        # We only submit new ensembles when there is not an active ensemble job
        self.futures: list[dask.distributed.Future] = []

        # The last criteria is the number of iterations
        self.iteration = 0

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()

    def __call__(
        self,
        smbo: "SMBO",
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> None:
        """
        Returns
        -------
        List[Tuple[int, float, float, float]]:
            A list with the performance history of this ensemble, of the form
            [(pandas_timestamp, train_performance, val_performance, test_performance)]
        """
        if result.status in (StatusType.STOP, StatusType.ABORT) or smbo._stop:
            return
        client = getattr(smbo.tae_runner, "client")
        self.build_ensemble(client)

    def build_ensemble(
        self,
        dask_client: dask.distributed.Client,
    ) -> None:
        """Build the ensemble

        Parameters
        ----------
        dask_client: dask.distributed.Client
            The dask client to use
        """
        # The second criteria is elapsed time
        elapsed_time = time.time() - self.start_time

        logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        # First test for termination conditions
        if self.time_left_for_ensembles < elapsed_time:
            logger.info(
                "Terminate ensemble building as not time is left (run for {}s)".format(
                    elapsed_time
                ),
            )
            return
        if self.max_iterations is not None and self.max_iterations <= self.iteration:
            logger.info(
                "Terminate ensemble building because of max iterations:"
                f" {self.max_iterations} of {self.iteration}"
            )
            return

        if len(self.futures) != 0:
            if self.futures[0].done():
                result = self.futures.pop().result()
                if result:
                    ensemble_history, self.ensemble_nbest, _, _, _ = result
                    logger.debug(
                        "iteration={} @ elapsed_time={} has history={}".format(
                            self.iteration,
                            elapsed_time,
                            ensemble_history,
                        )
                    )
                    self.history.extend(ensemble_history)

        # Only submit new jobs if the previous ensemble job finished
        if len(self.futures) == 0:

            # Add the result of the run
            # On the next while iteration, no references to
            # ensemble builder object, so it should be garbage collected to
            # save memory while waiting for resources
            # Also, notice how ensemble nbest is returned, so we don't waste
            # iterations testing if the deterministic predictions size can
            # be fitted in memory
            try:
                # Submit a Dask job from this job, to properly
                # see it in the dask diagnostic dashboard
                # Notice that the forked ensemble_builder_process will
                # wait for the below function to be done
                self.futures.append(
                    dask_client.submit(
                        fit_and_return_ensemble,
                        backend=self.backend,
                        dataset_name=self.dataset_name,
                        task_type=self.task,
                        metric=self.metric,
                        ensemble_size=self.ensemble_size,
                        ensemble_nbest=self.ensemble_nbest,
                        max_models_on_disc=self.max_models_on_disc,
                        seed=self.seed,
                        precision=self.precision,
                        memory_limit=self.ensemble_memory_limit,
                        read_at_most=self.read_at_most,
                        random_state=self.random_state,
                        end_at=self.start_time + self.time_left_for_ensembles,
                        iteration=self.iteration,
                        return_predictions=False,
                        priority=100,
                        pynisher_context=self.pynisher_context,
                        logger_port=self.logger_port,
                    )
                )

                logger.info(
                    "{}/{} Started Ensemble builder job at {} for iteration {}.".format(
                        # Log the client to make sure we
                        # remain connected to the scheduler
                        self.futures[0],
                        dask_client,
                        time.strftime("%Y.%m.%d-%H.%M.%S"),
                        self.iteration,
                    ),
                )
                self.iteration += 1
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)


def fit_and_return_ensemble(
    backend: Backend,
    dataset_name: str,
    task_type: int,
    metric: Scorer,
    ensemble_size: int,
    ensemble_nbest: int,
    seed: int,
    precision: int,
    read_at_most: int,
    end_at: float,
    iteration: int,
    return_predictions: bool,
    pynisher_context: str,
    max_models_on_disc: Optional[Union[float, int]] = 100,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    memory_limit: Optional[int] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> tuple[list[dict[str, Any]], int | float]:
    """

    A short function to fit and create an ensemble. It is just a wrapper to easily send
    a request to dask to create an ensemble and clean the memory when finished

    Parameters
    ----------
        backend: util.backend.Backend
            backend to write and read files

        dataset_name: str
            name of dataset

        metric: str
            name of metric to compute the loss of the given predictions

        task_type: int
            type of ML task

        ensemble_size: int
            maximal size of ensemble (passed to autosklearn.ensemble.ensemble_selection)

        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models

        max_models_on_disc: Optional[int | float] = 100
           Defines the maximum number of models that are kept in the disc.

           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.

           If float, it will be interpreted as the max megabytes allowed of disc space.
           That is, if the number of ensemble candidates require more disc space than
           this float value, the worst models will be deleted to keep within this
           budget. Models and predictions of the worst-performing models will be
           deleted then.

           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.

        seed: int
            random seed

        precision: [16,32,64,128]
            precision of floats to read the predictions

        read_at_most: int
            read at most n new prediction files in each iteration

        end_at: float
            At what time the job must finish. Needs to be the endtime and not the
            time left because we do not know when dask schedules the job.

        iteration: int
            The current iteration

        pynisher_context: str
            Context to use for multiprocessing, can be either fork, spawn or forkserver.

        logger_port: int = DEFAULT_TCP_LOGGING_PORT
            The port where the logging server is listening to.

        memory_limit: Optional[int] = None
            memory limit in mb. If ``None``, no memory limit is enforced.

        random_state: Optional[int | RandomState] = None
            A random state used for the ensemble selection process.

    Returns
    -------
    (ensemble_history: list[dict[str, Any]], nbest: int | float)
        The ensemble history and the nbest chosen members
    """
    result = EnsembleBuilder(
        backend=backend,
        dataset_name=dataset_name,
        task_type=task_type,
        metric=metric,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
        seed=seed,
        precision=precision,
        memory_limit=memory_limit,
        read_at_most=read_at_most,
        random_state=random_state,
        logger_port=logger_port,
    ).run(
        end_at=end_at,
        iteration=iteration,
        return_predictions=return_predictions,
        pynisher_context=pynisher_context,
    )
    return result
