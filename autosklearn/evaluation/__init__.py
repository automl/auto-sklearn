# -*- encoding: utf-8 -*-
from __future__ import absolute_import

import multiprocessing
import time

import pynisher
from smac.tae.execute_ta_run import StatusType

from .abstract_evaluator import *
from .cv_evaluator import *
from .holdout_evaluator import *
from .nested_cv_evaluator import *
from .test_evaluator import *
from .util import *


def eval_with_limits(datamanager, tmp_dir, config, seed, num_run,
                     resampling_strategy,
                     resampling_strategy_args, memory_limit,
                     func_eval_time_limit, subsample=None,
                     with_predictions=True,
                     all_scoring_functions=False,
                     output_y_test=True):
    if resampling_strategy_args is None:
        resampling_strategy_args = {}

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
    queue = multiprocessing.Queue()
    safe_eval = pynisher.enforce_limits(mem_in_mb=memory_limit,
                                        wall_time_in_s=func_eval_time_limit,
                                        cpu_time_in_s=func_eval_time_limit,
                                        grace_period_in_s=30)(
        eval_function)

    try:
        safe_eval(queue=queue, config=config, data=datamanager,
                  tmp_dir=tmp_dir, seed=seed, num_run=num_run,
                  subsample=subsample,
                  with_predictions=with_predictions,
                  all_scoring_functions=all_scoring_functions,
                  output_y_test=output_y_test,
                  **resampling_strategy_args)
        info = queue.get(block=True, timeout=1)
    except Exception as e0:
        if isinstance(e0, MemoryError):
            is_memory_error = True
        else:
            is_memory_error = False

        try:
            # This happens if a timeout is reached and a half-way trained
            #  model can be used to predict something
            info = queue.get_nowait()
        except Exception as e1:
            # This happens if a timeout is reached and the model does not
            #  support iterative_fit()
            duration = time.time() - start_time
            if is_memory_error:
                status = StatusType.MEMOUT
            elif duration >= func_eval_time_limit:
                status = StatusType.TIMEOUT
            else:
                status = StatusType.CRASHED
            info = (duration, 2.0, seed, str(e0), status)
    return info

