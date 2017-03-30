# -*- encoding: utf-8 -*-
import logging
import math
import multiprocessing

import pynisher
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit, KFold, \
    StratifiedKFold
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_func import AbstractTAFunc
from ConfigSpace import Configuration

from .abstract_evaluator import *
from .train_evaluator import *
from .test_evaluator import *
from .util import *
from autosklearn.constants import CLASSIFICATION_TASKS, MULTILABEL_CLASSIFICATION

WORST_POSSIBLE_RESULT = 1.0


# TODO potentially log all inputs to this class to pickle them in order to do
# easier debugging of potential crashes
class ExecuteTaFuncWithQueue(AbstractTAFunc):

    def __init__(self, backend, autosklearn_seed, resampling_strategy,
                 logger, initial_num_run=1, stats=None, runhistory=None,
                 run_obj='quality', par_factor=1, all_scoring_functions=False,
                 output_y_hat_optimization=True, include=None, exclude=None,
                 memory_limit=None, disable_file_output=False,
                 **resampling_strategy_args):

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
            output_y_hat_optimization = False
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
        self.all_scoring_functions = all_scoring_functions
        # TODO deactivate output_y_hat_optimization and let the respective evaluator decide
        self.output_y_hat_optimization = output_y_hat_optimization
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
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

        return super().start(config=config, instance=instance, cutoff=cutoff,
                             seed=seed, instance_specific=instance_specific,
                             capped=capped)

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific=None):

        D = self.backend.load_datamanager()
        queue = multiprocessing.Queue()

        if instance_specific is None or instance_specific == '0':
            instance_specific = {}
        else:
            instance_specific = [specific.split('=') for specific in instance_specific.split(',')]
            instance_specific = {specific[0]: specific[1] for specific in instance_specific}
        subsample = instance_specific.get('subsample')
        subsample = int(subsample) if subsample is not None else None

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
                          all_scoring_functions=self.all_scoring_functions,
                          output_y_hat_optimization=self.output_y_hat_optimization,
                          subsample=subsample,
                          include=self.include,
                          exclude=self.exclude,
                          disable_file_output=self.disable_file_output,
                          instance=instance)

        if self.resampling_strategy != 'test':
            cv = self.get_splitter(D)
            obj_kwargs['cv'] = cv
        #if instance is not None:
        #    obj_kwargs['instance'] = instance

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

    def get_splitter(self, D):
        y = D.data['Y_train'].ravel()
        n = D.data['Y_train'].shape[0]
        if D.info['task'] in CLASSIFICATION_TASKS and \
                        D.info['task'] != MULTILABEL_CLASSIFICATION:

            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:
                try:
                    cv = StratifiedShuffleSplit(y=y, n_iter=1, train_size=0.67,
                                                test_size=0.33, random_state=1)
                except ValueError as e:
                    if 'The least populated class in y has only' in e.args[0]:
                        cv = ShuffleSplit(n=n, n_iter=1, train_size=0.67,
                                          test_size=0.33, random_state=1)
                    else:
                        raise

            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                cv = StratifiedKFold(y=y,
                                     n_folds=self.resampling_strategy_args[
                                         'folds'],
                                     shuffle=True, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)
        else:
            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:
                cv = ShuffleSplit(n=n, n_iter=1, train_size=0.67,
                                  test_size=0.33, random_state=1)
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                cv = KFold(n=n,
                           n_folds=self.resampling_strategy_args['folds'],
                           shuffle=True, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)
        return cv

