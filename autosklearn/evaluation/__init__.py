# -*- encoding: utf-8 -*-
import logging
import multiprocessing
import sys
import time
import traceback

import pynisher
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_func import AbstractTAFunc

from .abstract_evaluator import *
from .cv_evaluator import *
from .holdout_evaluator import *
from .nested_cv_evaluator import *
from .test_evaluator import *
from .util import *

WORST_POSSIBLE_RESULT = 1.0


class ExecuteTaFuncWithQueue(AbstractTAFunc):

    def __init__(self, backend, autosklearn_seed, resampling_strategy,
                 logger, initial_num_run=1, stats=None, runhistory=None,
                 run_obj='quality', par_factor=1, with_predictions=True,
                 all_scoring_functions=False, output_y_test=True,
                 **resampling_strategy_args):

        if resampling_strategy == 'holdout':
            eval_function = eval_holdout
        elif resampling_strategy == 'holdout-iterative-fit':
            eval_function = eval_iterative_holdout
        elif resampling_strategy == 'cv':
            eval_function = eval_cv
        elif resampling_strategy == 'partial-cv':
            eval_function = eval_partial_cv
        elif resampling_strategy == 'test':
            eval_function = eval_t
            output_y_test = False
            with_predictions = False
        else:
            raise ValueError('Unknown resampling strategy %s' %
                             resampling_strategy)

        super().__init__(ta=eval_function, stats=stats, runhistory=runhistory,
                         run_obj=run_obj, par_factor=par_factor)

        self.backend = backend
        self.autosklearn_seed = autosklearn_seed
        self.resampling_strategy = resampling_strategy
        self.num_run = initial_num_run
        self.resampling_strategy_args = resampling_strategy_args
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions
        self.output_y_test = output_y_test
        self.logger = logger

    def run(self, config, instance=None,
            cutoff=None,
            memory_limit=None,
            seed=12345,
            instance_specific="0"):

        D = self.backend.load_datamanager()
        queue = multiprocessing.Queue()

        arguments = dict(logger=logging.getLogger("pynisher"),
                         wall_time_in_s=cutoff,
                         mem_in_mb=memory_limit)
        obj_kwargs = dict(queue=queue,
                          config=config,
                          data=D,
                          backend=self.backend,
                          seed=self.autosklearn_seed,
                          num_run=self.num_run,
                          with_predictions=self.with_predictions,
                          all_scoring_functions=self.all_scoring_functions,
                          output_y_test=self.output_y_test,
                          subsample=None,
                          **self.resampling_strategy_args)

        obj = pynisher.enforce_limits(**arguments)(self.ta)
        obj(**obj_kwargs)

        if obj.exit_status is pynisher.TimeoutException:
            status = StatusType.TIMEOUT
            cost = WORST_POSSIBLE_RESULT
            additional_run_info = 'Timeout'
        elif obj.exit_status is pynisher.MemorylimitException:
            status = StatusType.MEMOUT
            cost = WORST_POSSIBLE_RESULT
            additional_run_info = 'Memout'
        else:
            try:
                info = queue.get(block=True, timeout=2)
                result = info[1]
                error_message = info[3]

                if obj.exit_status == 0 and result is not None:
                    status = StatusType.SUCCESS
                    cost = result
                    additional_run_info = ''
                else:
                    status = StatusType.CRASHED
                    cost = WORST_POSSIBLE_RESULT
                    additional_run_info = error_message
            except Exception as e0:
                additional_run_info = 'Unknown error (%s) %s' % (type(e0), e0)
                status = StatusType.CRASHED
                cost = WORST_POSSIBLE_RESULT

        runtime = float(obj.wall_clock_time)
        self.num_run += 1
        return status, cost, runtime, additional_run_info

# def eval_with_limits(config, datamanager, backend, seed, num_run,
#                      resampling_strategy,
#                      resampling_strategy_args, memory_limit,
#                      func_eval_time_limit, subsample=None,
#                      with_predictions=True,
#                      all_scoring_functions=False,
#                      output_y_test=True,
#                      logger=None,
#                      # arguments to please SMAC
#                      instance=None):
#     if resampling_strategy_args is None:
#         resampling_strategy_args = {}
#
#     start_time = time.time()
#     queue = multiprocessing.Queue()
#     safe_eval = pynisher.enforce_limits(mem_in_mb=memory_limit,
#                                         wall_time_in_s=func_eval_time_limit,
#                                         grace_period_in_s=30,
#                                         logger=logger)(_eval_wrapper)
#
#     try:
#         safe_eval(queue=queue, config=config, data=datamanager,
#                   backend=backend, seed=seed, num_run=num_run,
#                   subsample=subsample,
#                   with_predictions=with_predictions,
#                   all_scoring_functions=all_scoring_functions,
#                   output_y_test=output_y_test,
#                   resampling_strategy=resampling_strategy,
#                   **resampling_strategy_args)
#         info = queue.get(block=True, timeout=2)
#
#     except Exception as e0:
#         error_message = 'Unknown error (%s) %s' % (type(e0), e0)
#         status = StatusType.CRASHED
#
#         duration = time.time() - start_time
#         info = (duration, WORST_POSSIBLE_RESULT, seed, error_message, status)
#
#     # TODO only return relevant information and make SMAC measure the rest!
#     # Currently, everything has the status SUCESS
#     #return info
#     return info[1], info[3]

