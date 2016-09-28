# -*- encoding: utf-8 -*-
from __future__ import absolute_import

import multiprocessing
import sys
import time
import traceback

import pynisher
from smac.tae.execute_ta_run import StatusType

from .abstract_evaluator import *
from .cv_evaluator import *
from .holdout_evaluator import *
from .nested_cv_evaluator import *
from .test_evaluator import *
from .util import *

WORST_POSSIBLE_RESULT = 2.0


def _eval_wrapper(queue, config, data, backend, seed, num_run, subsample,
                  with_predictions, all_scoring_functions, output_y_test,
                  resampling_strategy, **resampling_strategy_args):
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

    start_time = time.time()
    try:
        eval_function(queue=queue, config=config, data=data,
                      backend=backend, seed=seed, num_run=num_run,
                      subsample=subsample, with_predictions=with_predictions,
                      all_scoring_functions=all_scoring_functions,
                      output_y_test=output_y_test,
                      **resampling_strategy_args)
    # We need to catch the 'limit'-exceptions of the pynisher here as well!
    except pynisher.TimeoutException as e:
        duration = time.time() - start_time
        error_message = 'Timeout'
        queue.put((duration, WORST_POSSIBLE_RESULT, seed, error_message,
                   StatusType.TIMEOUT))
    except MemoryError as e:
        duration = time.time() - start_time
        error_message = 'Memout'
        queue.put((duration, WORST_POSSIBLE_RESULT, seed, error_message,
                   StatusType.MEMOUT))
    except Exception as e:
        duration = time.time() - start_time
        exc_info = sys.exc_info()
        error_message = ''.join(traceback.format_exception(*exc_info))
        queue.put((duration, WORST_POSSIBLE_RESULT, seed, error_message,
                   StatusType.CRASHED))


def eval_with_limits(datamanager, backend, config, seed, num_run,
                     resampling_strategy,
                     resampling_strategy_args, memory_limit,
                     func_eval_time_limit, subsample=None,
                     with_predictions=True,
                     all_scoring_functions=False,
                     output_y_test=True,
                     logger=None):
    if resampling_strategy_args is None:
        resampling_strategy_args = {}

    start_time = time.time()
    queue = multiprocessing.Queue()
    safe_eval = pynisher.enforce_limits(mem_in_mb=memory_limit,
                                        wall_time_in_s=func_eval_time_limit,
                                        grace_period_in_s=30,
                                        logger=logger)(_eval_wrapper)

    try:
        safe_eval(queue=queue, config=config, data=datamanager,
                  backend=backend, seed=seed, num_run=num_run,
                  subsample=subsample,
                  with_predictions=with_predictions,
                  all_scoring_functions=all_scoring_functions,
                  output_y_test=output_y_test,
                  resampling_strategy=resampling_strategy,
                  **resampling_strategy_args)
        info = queue.get(block=True, timeout=2)
    except Exception as e0:
        error_message = 'Unknown error (%s) %s' % (type(e0), e0)
        status = StatusType.CRASHED

        duration = time.time() - start_time
        info = (duration, WORST_POSSIBLE_RESULT, seed, error_message, status)

    return info

