from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union

import logging
import math
import multiprocessing
import time
import traceback
from queue import Empty

import pynisher
from ConfigSpace import Configuration
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.model_selection._split import _RepeatedSplits
from smac import Scenario
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.runner import TargetAlgorithmAbortException, TargetFunctionRunner

import autosklearn.evaluation.test_evaluator
import autosklearn.evaluation.train_evaluator
import autosklearn.pipeline.components
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.evaluation import test_evaluator, train_evaluator
from autosklearn.evaluation.util import empty_queue, extract_learning_curve, read_queue
from autosklearn.metrics import Scorer, get_cost_of_crash
from autosklearn.pipeline.components.base import _addons
from autosklearn.util.logging_ import PickableLoggerAdapter, get_named_client_logger
from autosklearn.util.parallel import preload_modules
from autosklearn.util.str_types import (
    BUDGET_TYPE,
    PYNISHER_CONTEXT,
    RESAMPLING_STRATEGY,
)


# TODO potentially log all inputs to this class to pickle them in order to do
#   easier debugging of potential crashes
class TargetFunctionRunnerWithQueue(TargetFunctionRunner):
    """The target function runner that takes into account that a trial can be killed
    prior to the model being fully fitted; thus putting intermediate results into a
    queue and querying them once the time is over.

    Parameters
    ----------
    scenario : Scenario
        The object describing the environment for the optimization.
    backend : Backend
        The backend that implements some common functionality.
    autosklearn_seed : int
        The seed for autosklearn.
    resampling_strategy : Union[RESAMPLING_STRATEGY, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit] # noqa E501
        A string or other resampling strategy types from sklearn based on which the
        target function is selected.
    metrics : Sequence[Scorer]
        A sequence of objects that host a function to calculate how good the
        prediction is according to the solution.
    port : int
        The port for the logger.
    pynisher_context : PYNISHER_CONTEXT
        Multiprocessing context of pynisher.
    initial_num_run : int = 1
        Starting number for the trials.
    scoring_functions : list[Scorer] | None = None
        A list of metrics to calculate multiple losses.
    output_y_hat_optimization : bool = True
        ?
    include : list[str] | None = None
        Specifies a step and the components that are included in search.
    exclude : list[str] | None = None
        Specifies a step and the components that are excluded from search.
    memory_limit : int | None = None
        Memory limit in MB for the trial.
    disable_file_output : bool = False
        Doesn't output to file if true.
    init_params : dict[str, Any] | None = None
        ?
    budget_type : BUDGET_TYPE | None = None
        ?
    **resampling_strategy_args : Any
        Arguments to pass in to the resampling strategy.
    """

    def __init__(
        self,
        scenario: Scenario,
        backend: Backend,
        autosklearn_seed: int,
        resampling_strategy: Union[
            RESAMPLING_STRATEGY, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
        ],
        metrics: Sequence[Scorer],
        port: int,
        pynisher_context: PYNISHER_CONTEXT,
        initial_num_run: int = 1,
        scoring_functions: list[Scorer] | None = None,
        output_y_hat_optimization: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        memory_limit: int | None = None,
        disable_file_output: bool = False,
        init_params: dict[str, Any] | None = None,
        budget_type: BUDGET_TYPE | None = None,
        **resampling_strategy_args: Any,
    ):
        self.worst_possible_result = get_cost_of_crash(metrics)
        self.backend = backend
        self.autosklearn_seed = autosklearn_seed
        self.initial_num_run = initial_num_run
        self.metrics = metrics
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.scoring_functions = scoring_functions
        self.output_y_hat_optimization = output_y_hat_optimization
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.init_params = init_params
        self.budget_type = budget_type
        self.output_y_hat_optimization = output_y_hat_optimization

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        dm = self.backend.load_datamanager()
        self._get_test_loss = "X_test" in dm.data and "Y_test" in dm.data

        self.port = port
        self.pynisher_context = pynisher_context
        if self.port is None:
            self.logger: Union[
                logging.Logger, PickableLoggerAdapter
            ] = logging.getLogger("TAE")
        else:
            self.logger = get_named_client_logger(
                name="TAE",
                port=self.port,
            )

        target_function = self._select_target_function(resampling_strategy)

        target_function_wrapper = TargetFunctionWrapper(
            target_function=target_function,
            worst_possible_result=self.worst_possible_result,
        )

        super().__init__(
            scenario=scenario,
            target_function=target_function_wrapper.target_function,
        )

    def run_wrapper(
        self,
        trial_info: TrialInfo,
    ) -> Tuple[TrialInfo, TrialValue]:
        """Wraps run() to cap the target algorithm runtime. It is called by the parent's
        submit_trial() method.

        Parameters
        ----------
        trial_info : TrialInfo
            Object that contains enough information to execute a configuration run in
            isolation.

        Returns
        -------
        TrialInfo:
            An object containing the configuration launched.
        TrialValue:
            Contains information about the status/performance of the ran configuration.
        """
        # updatesmac: config_id is None, how?
        # config_id = (
        #    trial_info.config
        #    if isinstance(trial_info.config, int)
        #    else trial_info.config.config_id
        # )
        # self.logger.info("Starting to evaluate configuration %d" % config_id)

        return super().run_wrapper(trial_info=trial_info)

    # updatesmac: change function signature to accepting a TrialInfo and returning a
    #  TrialValue
    def run(
        self,
        config: Configuration | int,
        instance: str | None = None,
        budget: float | None = 0.0,
        seed: int | None = 12345,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Runs the target function in a queue.

        It essentially translates a TrialInfo to a TrialValue.

        Parameters
        ----------
        config : Configuration
            Specifies the configurations of the components of the pipeline that
            should be run. When an integer value, then it treats it as a dummy
            configuration.
        instance : str
            A string identifier for a subset of the dataset that the configuration
            will run on. When None, then it will run on the whole dataset.
        budget : float
            Represents a limit to running the target function. This is handled by the
            target function internally.
        seed : int
            The seed for the run.

        Returns
        -------
        trial_value : tuple[StatusType, float | list[float], float, dict]
            Includes useful statistics about the ran trial. For example, the cost that
            the pipeline with the given configuration achieved on the dataset.
        """
        init_params = {"instance": instance}
        if self.init_params is not None:
            init_params.update(self.init_params)

        memory_limit = None
        wall_time_limit = None
        if self.memory_limit is not None:
            memory_limit = (self.memory_limit, "MB")

        if self._scenario.trial_walltime_limit is not None:
            wall_time_limit = (self._scenario.trial_walltime_limit, "s")

        # Dummy config when `isinstance(config, int)`
        if isinstance(config, int):
            num_run = self.initial_num_run
        else:
            num_run = config.config_id + self.initial_num_run

        context = multiprocessing.get_context(self.pynisher_context)
        queue = context.Queue()

        if context == "forkserver":
            preload_modules(context)

        obj_kwargs = dict(
            queue=queue,
            config=config,
            instance=instance,
            init_params=init_params,
            budget=budget,
            num_run=num_run,
            additional_components=_addons,
            backend=self.backend,
            port=self.port,
            metrics=self.metrics,
            seed=self.autosklearn_seed,
            output_y_hat_optimization=self.output_y_hat_optimization,
            scoring_functions=self.scoring_functions,
            include=self.include,
            exclude=self.exclude,
            disable_file_output=self.disable_file_output,
            budget_type=self.budget_type,
        )

        # This seems to be used scripts/run_auto-sklearn_for_metadata_generation.py
        if self.resampling_strategy != "test":
            obj_kwargs["resampling_strategy"] = self.resampling_strategy
            obj_kwargs["resampling_strategy_args"] = self.resampling_strategy_args

        # Wrap the target function into pynisher that limits the maximum resources
        # that a call to this function can take
        target_function = pynisher.limit(
            self._target_function,
            memory=memory_limit,
            wall_time=wall_time_limit,
            context=self.pynisher_context,
        )

        start = time.time()
        try:
            # Run the target function and put its result into a queue
            target_function(**obj_kwargs)
            error = None
            _traceback = None
        except Exception as e:
            error = e
            _traceback = traceback.format_exc()
            print(error)

        duration = time.time() - start

        cost = self.worst_possible_result
        additional_run_info = {}
        status = None
        info = None
        # We now try read out and see if we got a result from the queue
        try:
            info = read_queue(queue)
            cost = info[-1].get("loss", self.worst_possible_result)
            status = info[-1].get("status")
            additional_run_info = info[-1].get("additional_run_info", {})
        except Empty:
            pass
        finally:
            if status is None:
                if error is None:
                    status = StatusType.CRASHED
                elif isinstance(error, MemoryError):
                    status = StatusType.MEMORYOUT
                elif isinstance(error, pynisher.TimeoutException):
                    status = StatusType.TIMEOUT
                elif isinstance(error, TargetAlgorithmAbortException):
                    status = StatusType.TIMEOUT
                else:
                    status = StatusType.CRASHED

            if error is not None:
                additional_run_info["error"] = repr(error)

            if _traceback is not None:
                additional_run_info["traceback"] = _traceback

        if not isinstance(additional_run_info, dict):
            additional_run_info = {"message": additional_run_info}

        if (
            info is not None
            and self.resampling_strategy
            in ("holdout-iterative-fit", "cv-iterative-fit")
            and status != StatusType.CRASHED
        ):
            learning_curve = extract_learning_curve(info)
            learning_curve_runtime = extract_learning_curve(info, "duration")
            if len(learning_curve) > 1:
                additional_run_info["learning_curve"] = learning_curve
                additional_run_info["learning_curve_runtime"] = learning_curve_runtime

            train_learning_curve = extract_learning_curve(info, "train_loss")
            if len(train_learning_curve) > 1:
                additional_run_info["train_learning_curve"] = train_learning_curve
                additional_run_info["learning_curve_runtime"] = learning_curve_runtime

            if self._get_test_loss:
                test_learning_curve = extract_learning_curve(info, "test_loss")
                if len(test_learning_curve) > 1:
                    additional_run_info["test_learning_curve"] = test_learning_curve
                    additional_run_info[
                        "learning_curve_runtime"
                    ] = learning_curve_runtime

        if isinstance(config, int):
            origin = "DUMMY"
            config_id = config
        else:
            origin = getattr(config, "origin", "UNKNOWN")
            # updatesmac: why is config_id marked as unkown attribute? it exists
            config_id = config.config_id

        additional_run_info["configuration_origin"] = origin

        empty_queue(queue)
        self.logger.info("Finished evaluating configuration %d" % config_id)

        # Do some sanity checking (for multi objective)
        if self._n_objectives > 1:
            error = (
                f"Returned costs {cost} does not match the number of objectives"
                f" {self._n_objectives}."
            )

            # If dict convert to array
            # Make sure the ordering is correct
            if isinstance(cost, dict):
                ordered_cost = []
                for name in self._objectives:
                    if name not in cost:
                        raise RuntimeError(
                            f"Objective {name} was not found "
                            f"in the returned costs ({cost})"
                        )

                    ordered_cost.append(cost[name])
                cost = ordered_cost

            if isinstance(cost, list):
                if len(cost) != self._n_objectives:
                    raise RuntimeError(error)

            if isinstance(cost, float):
                raise RuntimeError(error)

        return status, cost, float(duration), additional_run_info

    @staticmethod
    def _select_target_function(
        resampling_strategy: Union[
            RESAMPLING_STRATEGY, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
        ]
    ) -> Callable:
        """Returns the function that should be optimized based on the resampling
        strategy.
        """
        if resampling_strategy == "holdout":
            target_function = train_evaluator.eval_holdout
        elif resampling_strategy == "holdout-iterative-fit":
            target_function = train_evaluator.eval_iterative_holdout
        elif resampling_strategy == "cv-iterative-fit":
            target_function = train_evaluator.eval_iterative_cv
        elif resampling_strategy == "cv" or isinstance(
            resampling_strategy, (BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit)
        ):
            target_function = train_evaluator.eval_cv
        elif resampling_strategy == "partial-cv":
            target_function = autosklearn.evaluation.train_evaluator.eval_partial_cv
        elif resampling_strategy == "partial-cv-iterative-fit":
            target_function = train_evaluator.eval_partial_cv_iterative
        elif resampling_strategy == "test":
            target_function = test_evaluator.eval_t
        else:
            raise ValueError("Unknown resampling strategy %s" % resampling_strategy)

        return target_function


class TargetFunctionWrapper:
    """Wrapper class around the target function to make the target function runner
    class easier to understand, and to avoid the problem with partial functions."""

    def __init__(
        self, target_function: Callable, worst_possible_result: list[float] | float
    ):
        self._target_function = target_function
        self._worst_possible_result = worst_possible_result

    def target_function(self, queue: multiprocessing.Queue, **kwargs: Any) -> None:
        """A wrapper around the target function to handle exceptions in case the
        queue doesn't work,

        which happened with the following traceback:
             File "auto-sklearn/autosklearn/evaluation/__init__.py", line 29, in fit_predict_try_except_decorator  # noqa E501
             return ta(queue=queue, **kwargs)
             File "auto-sklearn/autosklearn/evaluation/train_evaluator.py", line 1067, in eval_holdout  # noqa E501
             evaluator.fit_predict_and_loss(iterative=iterative)
             File "auto-sklearn/autosklearn/evaluation/train_evaluator.py", line 616, in fit_predict_and_loss,  # noqa E501
             status=status
             File "auto-sklearn/autosklearn/evaluation/abstract_evaluator.py", line 320, in finish_up  # noqa E501
             self.queue.put(return_value_dict)
             File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/multiprocessing/queues.py", line 87, in put  # noqa E501
             self._start_thread()
             File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/multiprocessing/queues.py", line 170, in _start_thread  # noqa E501
             self._thread.start()
             File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/threading.py", line 847, in start  # noqa E501
             RuntimeError: can't start new thread

        This function used to be called fit_predict_try_except_decorator.

        Note: Using an instance function instead of a partial function, because SMAC
        calls __code__ on the target function and 'functools.partial' object has no
        attribute '__code__'

        Parameters
        ----------
        queue : Queue
            The multiprocessing queue
        kwargs : Any
            See any of the eval functions in autosklearn.evaluation.train_evaluator
            for remaining keyword arguments. For example, eval_holdout.

        Returns
        -------

        """
        try:
            return self._target_function(queue=queue, **kwargs)
        except Exception as e:
            if isinstance(e, (MemoryError, pynisher.TimeoutException)):
                # Re-raise the memory error to let the pynisher handle that correctly
                raise e

            exception_traceback = traceback.format_exc()
            error_message = repr(e)

            print(
                "Exception handling in `fit_predict_try_except_decorator`: "
                "traceback: %s \nerror message: %s"
                % (exception_traceback, error_message)
            )

            queue.put(
                {
                    "loss": self.worst_possible_result,
                    "additional_run_info": {
                        "traceback": exception_traceback,
                        "error": error_message,
                    },
                    "status": StatusType.CRASHED,
                    "final_queue_element": True,
                },
                block=True,
            )
            queue.close()
