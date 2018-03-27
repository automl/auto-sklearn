# -*- encoding: utf-8 -*-
import functools
import logging
import math
import multiprocessing
from queue import Empty
import traceback

import pynisher
from smac.tae.execute_ta_run import StatusType, BudgetExhaustedException, \
    TAEAbortException
from smac.tae.execute_func import AbstractTAFunc
from ConfigSpace import Configuration
from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit,\
    BaseCrossValidator

import autosklearn.evaluation.train_evaluator
import autosklearn.evaluation.test_evaluator
import autosklearn.evaluation.util

WORST_POSSIBLE_RESULT = 1.0


def fit_predict_try_except_decorator(ta, queue, **kwargs):

    try:
        return ta(queue=queue, **kwargs)
    except Exception as e:
        if isinstance(e, MemoryError):
            # Re-raise the memory error to let the pynisher handle that
            # correctly
            raise e

        exception_traceback = traceback.format_exc()
        error_message = repr(e)

        queue.put({'loss': WORST_POSSIBLE_RESULT,
                   'additional_run_info': {'traceback': exception_traceback,
                                           'error': error_message},
                   'status': StatusType.CRASHED,
                   'final_queue_element': True})


# TODO potentially log all inputs to this class to pickle them in order to do
# easier debugging of potential crashes
class ExecuteTaFuncWithQueue(AbstractTAFunc):

    def __init__(self, backend, autosklearn_seed, resampling_strategy, metric,
                 logger, initial_num_run=1, stats=None, runhistory=None,
                 run_obj='quality', par_factor=1, all_scoring_functions=False,
                 output_y_hat_optimization=True, include=None, exclude=None,
                 memory_limit=None, disable_file_output=False, init_params=None,
                 **resampling_strategy_args):

        if resampling_strategy == 'holdout':
            eval_function = autosklearn.evaluation.train_evaluator.eval_holdout
        elif resampling_strategy == 'holdout-iterative-fit':
            eval_function = autosklearn.evaluation.train_evaluator.eval_iterative_holdout
        elif resampling_strategy == 'cv' or \
                issubclass(resampling_strategy, BaseCrossValidator) or \
                issubclass(resampling_strategy, _RepeatedSplits) or \
                issubclass(resampling_strategy, BaseShuffleSplit):
            eval_function = autosklearn.evaluation.train_evaluator.eval_cv
        elif resampling_strategy == 'partial-cv':
            eval_function = autosklearn.evaluation.train_evaluator.eval_partial_cv
        elif resampling_strategy == 'partial-cv-iterative-fit':
            eval_function = autosklearn.evaluation.train_evaluator.eval_partial_cv_iterative
        elif resampling_strategy == 'test':
            eval_function = autosklearn.evaluation.test_evaluator.eval_t
            output_y_hat_optimization = False
        else:
            raise ValueError('Unknown resampling strategy %s' %
                             resampling_strategy)

        eval_function = functools.partial(fit_predict_try_except_decorator,
                                          ta=eval_function)
        super().__init__(
            ta=eval_function,
            stats=stats,
            runhistory=runhistory,
            run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=WORST_POSSIBLE_RESULT,
        )

        self.backend = backend
        self.autosklearn_seed = autosklearn_seed
        self.resampling_strategy = resampling_strategy
        self.num_run = initial_num_run
        self.metric = metric
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.all_scoring_functions = all_scoring_functions
        # TODO deactivate output_y_hat_optimization and let the respective evaluator decide
        self.output_y_hat_optimization = output_y_hat_optimization
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.init_params = init_params
        self.logger = logger

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

    def start(self, config: Configuration,
              instance: str,
              cutoff: float = None,
              seed: int = 12345,
              instance_specific: str = "0",
              capped: bool = False):
        """
        wrapper function for ExecuteTARun.start() to cap the target algorithm
        runtime if it would run over the total allowed runtime.

        Parameters
        ----------
            config : Configuration
                mainly a dictionary param -> value
            instance : string
                problem instance
            cutoff : float
                runtime cutoff
            seed : int
                random seed
            instance_specific: str
                instance specific information (e.g., domain file or solution)
            capped: bool
                if true and status is StatusType.TIMEOUT,
                uses StatusType.CAPPED
        Returns
        -------
            status: enum of StatusType (int)
                {SUCCESS, TIMEOUT, CRASHED, ABORT}
            cost: float
                cost/regret/quality (float) (None, if not returned by TA)
            runtime: float
                runtime (None if not returned by TA)
            additional_info: dict
                all further additional run information
        """
        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < cutoff:
            cutoff = int(remaining_time - 5)

        if cutoff <= 1.0:
            raise BudgetExhaustedException()
        cutoff = int(cutoff)

        return super().start(config=config, instance=instance, cutoff=cutoff,
                             seed=seed, instance_specific=instance_specific,
                             capped=capped)

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific=None):

        queue = multiprocessing.Queue()

        if not (instance_specific is None or instance_specific == '0'):
            raise ValueError(instance_specific)
        init_params = {'instance': instance}
        if self.init_params is not None:
            init_params.update(self.init_params)

        arguments = dict(logger=logging.getLogger("pynisher"),
                         wall_time_in_s=cutoff,
                         mem_in_mb=self.memory_limit)
        obj_kwargs = dict(queue=queue,
                          config=config,
                          backend=self.backend,
                          metric=self.metric,
                          seed=self.autosklearn_seed,
                          num_run=self.num_run,
                          all_scoring_functions=self.all_scoring_functions,
                          output_y_hat_optimization=self.output_y_hat_optimization,
                          include=self.include,
                          exclude=self.exclude,
                          disable_file_output=self.disable_file_output,
                          instance=instance,
                          init_params=init_params)

        if self.resampling_strategy != 'test':
            obj_kwargs['resampling_strategy'] = self.resampling_strategy
            obj_kwargs['resampling_strategy_args'] = self.resampling_strategy_args
        #if instance is not None:
        #    obj_kwargs['instance'] = instance

        obj = pynisher.enforce_limits(**arguments)(self.ta)
        obj(**obj_kwargs)

        if obj.exit_status is pynisher.TimeoutException:
            # Even if the pynisher thinks that a timeout occured, it can be that
            # the target algorithm wrote something into the queue - then we
            # treat it as a succesful run
            try:
                info = autosklearn.evaluation.util.get_last_result(queue)
                result = info['loss']
                status = info['status']
                additional_run_info = info['additional_run_info']
                additional_run_info['info'] = 'Run stopped because of timeout.'

                if status == StatusType.SUCCESS:
                    cost = result
                else:
                    cost = WORST_POSSIBLE_RESULT

            except Empty:
                status = StatusType.TIMEOUT
                cost = WORST_POSSIBLE_RESULT
                additional_run_info = {'error': 'Timeout'}

        elif obj.exit_status is pynisher.MemorylimitException:
            status = StatusType.MEMOUT
            cost = WORST_POSSIBLE_RESULT
            additional_run_info = {'error': 'Memout (used more than %d MB).' %
                                            self.memory_limit}

        elif obj.exit_status is TAEAbortException:
            status = StatusType.ABORT
            cost = WORST_POSSIBLE_RESULT
            additional_run_info = {'error': 'Your configuration of '
                                            'auto-sklearn does not work!'}

        else:
            try:
                info = autosklearn.evaluation.util.get_last_result(queue)
                result = info['loss']
                status = info['status']
                additional_run_info = info['additional_run_info']

                if obj.exit_status == 0:
                    cost = result
                else:
                    status = StatusType.CRASHED
                    cost = WORST_POSSIBLE_RESULT
                    additional_run_info['info'] = 'Run treated as crashed ' \
                                                  'because the pynisher exit ' \
                                                  'status %s is unknown.' % \
                                                  str(obj.exit_status)
            except Empty:
                additional_run_info = {'error': 'Result queue is empty'}
                status = StatusType.CRASHED
                cost = WORST_POSSIBLE_RESULT

        if not isinstance(additional_run_info, dict):
            additional_run_info = {'message': additional_run_info}

        if isinstance(config, int):
            origin = 'DUMMY'
        else:
            origin = getattr(config, 'origin', 'UNKNOWN')
        additional_run_info['configuration_origin'] = origin

        runtime = float(obj.wall_clock_time)
        self.num_run += 1

        autosklearn.evaluation.util.empty_queue(queue)

        return status, cost, runtime, additional_run_info



