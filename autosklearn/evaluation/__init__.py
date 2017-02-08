# -*- encoding: utf-8 -*-
import logging
import math
import multiprocessing

import numpy as np
import pynisher
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit, KFold, \
    StratifiedKFold
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_func import AbstractTAFunc

from .abstract_evaluator import *
from .train_evaluator import *
from .test_evaluator import *
from .util import *
from autosklearn.constants import REGRESSION_TASKS, CLASSIFICATION_TASKS, \
    MULTILABEL_CLASSIFICATION

WORST_POSSIBLE_RESULT = 1.0


# TODO potentially log all inputs to this class to pickle them in order to do
# easier debugging of potential crashes
class ExecuteTaFuncWithQueue(AbstractTAFunc):

    def __init__(self, backend, autosklearn_seed, resampling_strategy,
                 logger, initial_num_run=1, stats=None, runhistory=None,
                 run_obj='quality', par_factor=1, with_predictions=True,
                 all_scoring_functions=False, output_y_test=True,
                 include=None, exclude=None, memory_limit=None,
                 disable_file_output=False, **resampling_strategy_args):

        if resampling_strategy == 'holdout':
            eval_function = eval_holdout
        elif resampling_strategy == 'holdout-iterative-fit':
            eval_function = eval_iterative_holdout
        elif resampling_strategy == 'cv':
            eval_function = eval_cv
        elif resampling_strategy == 'partial-cv':
            eval_function = eval_partial_cv
        elif resampling_strategy == 'partial-cv-iterative-fit':
            eval_function = eval_partial_cv_iterative
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
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions
        # TODO deactivate output_y_test and let the respective evaluator decide
        self.output_y_test = output_y_test
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.logger = logger

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

    def start(self, config, instance,
              cutoff=None,
              seed=12345,
              instance_specific="0"):
        # Overwrite the start function here. This allows us to abort target
        # algorithm runs if the time us over without having the start method
        # of the parent class adding the run to the runhistory

        # Restrict the cutoff to not go over the final time limit, but stop ten
        # seconds earlier
        remaining_time = self.stats.get_remaing_time_budget()
        if remaining_time - 5 < cutoff:
            cutoff = int(remaining_time - 5)

        if cutoff <= 0:
            self.logger.debug(
                "Skip target algorithm run due to exhausted configuration budget")
            return StatusType.ABORT, np.nan, 0, {"misc": "exhausted bugdet -- ABORT"}

        return super().start(config=config, instance=instance, cutoff=cutoff,
                             seed=seed, instance_specific=instance_specific)

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific="0"):

        D = self.backend.load_datamanager()
        queue = multiprocessing.Queue()

        if D.info['task'] in CLASSIFICATION_TASKS and \
                D.info['task'] != MULTILABEL_CLASSIFICATION:
            y = D.data['Y_train'].ravel()
            if self.resampling_strategy in ['holdout', 'holdout-iterative-fit']:
                cv = StratifiedShuffleSplit(y=y, n_iter=1, train_size=0.67,
                                            test_size=0.33, random_state=1)
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                cv = StratifiedKFold(y=y, n_folds=self.resampling_strategy_args['folds'],
                                     shuffle=True, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)
        else:
            n = D.data['Y_train'].shape[0]
            if self.resampling_strategy in ['holdout', 'holdout-iterative-fit']:
                cv = ShuffleSplit(n=n, n_iter=1, train_size=0.67,
                                  test_size=0.33, random_state=1)
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                cv = KFold(n=n, n_folds=self.resampling_strategy_args['folds'],
                           shuffle=True, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)

        arguments = dict(logger=logging.getLogger("pynisher"),
                         wall_time_in_s=cutoff,
                         mem_in_mb=self.memory_limit,
                         grace_period_in_s=15)
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
                          include=self.include,
                          exclude=self.exclude,
                          disable_file_output=self.disable_file_output)
        if self.resampling_strategy != 'test':
            obj_kwargs['cv'] = cv
        if instance is not None:
            obj_kwargs['instance'] = instance

        obj = pynisher.enforce_limits(**arguments)(self.ta)
        obj(**obj_kwargs)

        if obj.exit_status is pynisher.TimeoutException:
            # Even if the pynisher thinks that a timeout occured, it can be that
            # the target algorithm wrote something into the queue - then we
            # treat it as a succesful run
            try:
                info = get_last_result(queue)
                result = info[1]
                additional_run_info = info[3]

                if obj.exit_status == pynisher.TimeoutException and result is not None:
                    status = StatusType.SUCCESS
                    cost = result
                else:
                    status = StatusType.CRASHED
                    cost = WORST_POSSIBLE_RESULT
            except Exception:
                status = StatusType.TIMEOUT
                cost = WORST_POSSIBLE_RESULT
                additional_run_info = 'Timeout'

        elif obj.exit_status is pynisher.MemorylimitException:
            status = StatusType.MEMOUT
            cost = WORST_POSSIBLE_RESULT
            additional_run_info = 'Memout'
        else:
            try:
                info = get_last_result(queue)
                result = info[1]
                additional_run_info = info[3]

                if obj.exit_status == 0 and result is not None:
                    status = StatusType.SUCCESS
                    cost = result
                else:
                    status = StatusType.CRASHED
                    cost = WORST_POSSIBLE_RESULT
            except Exception as e0:
                additional_run_info = 'Unknown error (%s) %s' % (type(e0), e0)
                status = StatusType.CRASHED
                cost = WORST_POSSIBLE_RESULT

        runtime = float(obj.wall_clock_time)
        self.num_run += 1
        return status, cost, runtime, additional_run_info

