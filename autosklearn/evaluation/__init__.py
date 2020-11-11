# -*- encoding: utf-8 -*-
import functools
import json
import math
import multiprocessing
from queue import Empty
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
import pynisher
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType, TAEAbortException
from smac.tae.execute_func import AbstractTAFunc

from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit,\
    BaseCrossValidator
from autosklearn.metrics import Scorer

import autosklearn.evaluation.train_evaluator
import autosklearn.evaluation.test_evaluator
import autosklearn.evaluation.util
import autosklearn.util.logging_


def fit_predict_try_except_decorator(ta, queue, cost_for_crash, **kwargs):

    try:
        return ta(queue=queue, **kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise e

        exception_traceback = traceback.format_exc()
        error_message = repr(e)

        # Printing stuff to stdout just in case the queue doesn't work, which happened with the
        # following traceback:
        #     File "auto-sklearn/autosklearn/evaluation/__init__.py", line 29, in fit_predict_try_except_decorator  # noqa E501
        #     return ta(queue=queue, **kwargs)
        #     File "auto-sklearn/autosklearn/evaluation/train_evaluator.py", line 1067, in eval_holdout  # noqa E501
        #     evaluator.fit_predict_and_loss(iterative=iterative)
        #     File "auto-sklearn/autosklearn/evaluation/train_evaluator.py", line 616, in fit_predict_and_loss,  # noqa E501
        #     status=status
        #     File "auto-sklearn/autosklearn/evaluation/abstract_evaluator.py", line 320, in finish_up  # noqa E501
        #     self.queue.put(rval_dict)
        #     File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/multiprocessing/queues.py", line 87, in put  # noqa E501
        #     self._start_thread()
        #     File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/multiprocessing/queues.py", line 170, in _start_thread  # noqa E501
        #     self._thread.start()
        #     File "miniconda/3-4.5.4/envs/autosklearn/lib/python3.7/threading.py", line 847, in start  # noqa E501
        #     RuntimeError: can't start new thread
        print("Exception handling in `fit_predict_try_except_decorator`: "
              "traceback: %s \nerror message: %s" % (exception_traceback, error_message))

        queue.put({'loss': cost_for_crash,
                   'additional_run_info': {'traceback': exception_traceback,
                                           'error': error_message},
                   'status': StatusType.CRASHED,
                   'final_queue_element': True}, block=True)
        queue.close()


def get_cost_of_crash(metric):

    # The metric must always be defined to extract optimum/worst
    if not isinstance(metric, Scorer):
        raise ValueError("The metric must be stricly be an instance of Scorer")

    # Autosklearn optimizes the err. This function translates
    # worst_possible_result to be a minimization problem.
    # For metrics like accuracy that are bounded to [0,1]
    # metric.optimum==1 is the worst cost.
    # A simple guide is to use greater_is_better embedded as sign
    if metric._sign < 0:
        worst_possible_result = metric._worst_possible_result
    else:
        worst_possible_result = metric._optimum - metric._worst_possible_result

    return worst_possible_result


def _encode_exit_status(exit_status):
    try:
        json.dumps(exit_status)
        return exit_status
    except (TypeError, OverflowError):
        return str(exit_status)


# TODO potentially log all inputs to this class to pickle them in order to do
# easier debugging of potential crashes
class ExecuteTaFuncWithQueue(AbstractTAFunc):

    def __init__(self, backend, autosklearn_seed, resampling_strategy, metric,
                 logger, cost_for_crash, abort_on_first_run_crash,
                 initial_num_run=1, stats=None,
                 run_obj='quality', par_factor=1, all_scoring_functions=False,
                 output_y_hat_optimization=True, include=None, exclude=None,
                 memory_limit=None, disable_file_output=False, init_params=None,
                 budget_type=None, ta=False, **resampling_strategy_args):

        if resampling_strategy == 'holdout':
            eval_function = autosklearn.evaluation.train_evaluator.eval_holdout
        elif resampling_strategy == 'holdout-iterative-fit':
            eval_function = autosklearn.evaluation.train_evaluator.eval_iterative_holdout
        elif resampling_strategy == 'cv-iterative-fit':
            eval_function = autosklearn.evaluation.train_evaluator.eval_iterative_cv
        elif resampling_strategy == 'cv' or (
             isinstance(resampling_strategy, type) and (
                issubclass(resampling_strategy, BaseCrossValidator) or
                issubclass(resampling_strategy, _RepeatedSplits) or
                issubclass(resampling_strategy, BaseShuffleSplit)
                )
             ):
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

        self.worst_possible_result = cost_for_crash

        eval_function = functools.partial(
            fit_predict_try_except_decorator,
            ta=eval_function,
            cost_for_crash=self.worst_possible_result,
        )

        super().__init__(
            ta=eval_function,
            stats=stats,
            run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=self.worst_possible_result,
            abort_on_first_run_crash=abort_on_first_run_crash,
        )

        self.backend = backend
        self.autosklearn_seed = autosklearn_seed
        self.resampling_strategy = resampling_strategy
        self.initial_num_run = initial_num_run
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
        self.budget_type = budget_type
        self.logger = logger

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        dm = self.backend.load_datamanager()
        if 'X_valid' in dm.data and 'Y_valid' in dm.data:
            self._get_validation_loss = True
        else:
            self._get_validation_loss = False
        if 'X_test' in dm.data and 'Y_test' in dm.data:
            self._get_test_loss = True
        else:
            self._get_test_loss = False

    def run_wrapper(
        self,
        run_info: RunInfo,
    ) -> Tuple[RunInfo, RunValue]:
        """
        wrapper function for ExecuteTARun.run_wrapper() to cap the target algorithm
        runtime if it would run over the total allowed runtime.

        Parameters
        ----------
        run_info : RunInfo
            Object that contains enough information to execute a configuration run in
            isolation.
        Returns
        -------
        RunInfo:
            an object containing the configuration launched
        RunValue:
            Contains information about the status/performance of config
        """
        if self.budget_type is None:
            if run_info.budget != 0:
                raise ValueError(
                    'If budget_type is None, budget must be.0, but is %f' % run_info.budget
                )
        else:
            if run_info.budget == 0:
                run_info = run_info._replace(budget=100)
            elif run_info.budget <= 0 or run_info.budget > 100:
                raise ValueError('Illegal value for budget, must be >0 and <=100, but is %f' %
                                 run_info.budget)
            if self.budget_type not in ('subsample', 'iterations', 'mixed'):
                raise ValueError("Illegal value for budget type, must be one of "
                                 "('subsample', 'iterations', 'mixed'), but is : %s" %
                                 self.budget_type)

        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < run_info.cutoff:
            run_info = run_info._replace(cutoff=int(remaining_time - 5))

        if run_info.cutoff < 1.0:
            return run_info, RunValue(
                status=StatusType.STOP,
                cost=self.worst_possible_result,
                time=0.0,
                additional_info={},
                starttime=time.time(),
                endtime=time.time(),
            )
        elif (
            run_info.cutoff != int(np.ceil(run_info.cutoff))
            and not isinstance(run_info.cutoff, int)
        ):
            run_info = run_info._replace(cutoff=int(np.ceil(run_info.cutoff)))

        return super().run_wrapper(run_info=run_info)

    def run(
        self,
        config: Configuration,
        instance: Optional[str] = None,
        cutoff: Optional[float] = None,
        seed: int = 12345,
        budget: float = 0.0,
        instance_specific: Optional[str] = None,
    ) -> Tuple[StatusType, float, float, Dict[str, Union[int, float, str, Dict, List, Tuple]]]:

        queue = multiprocessing.Queue()

        if not (instance_specific is None or instance_specific == '0'):
            raise ValueError(instance_specific)
        init_params = {'instance': instance}
        if self.init_params is not None:
            init_params.update(self.init_params)

        arguments = dict(
            logger=autosklearn.util.logging_.get_logger("pynisher"),
            wall_time_in_s=cutoff,
            mem_in_mb=self.memory_limit,
            capture_output=True,
        )

        if isinstance(config, int):
            num_run = self.initial_num_run
        else:
            num_run = config.config_id + self.initial_num_run

        obj_kwargs = dict(
            queue=queue,
            config=config,
            backend=self.backend,
            metric=self.metric,
            seed=self.autosklearn_seed,
            num_run=num_run,
            all_scoring_functions=self.all_scoring_functions,
            output_y_hat_optimization=self.output_y_hat_optimization,
            include=self.include,
            exclude=self.exclude,
            disable_file_output=self.disable_file_output,
            instance=instance,
            init_params=init_params,
            budget=budget,
            budget_type=self.budget_type,
        )

        if self.resampling_strategy != 'test':
            obj_kwargs['resampling_strategy'] = self.resampling_strategy
            obj_kwargs['resampling_strategy_args'] = self.resampling_strategy_args

        try:
            obj = pynisher.enforce_limits(**arguments)(self.ta)
            obj(**obj_kwargs)
        except Exception as e:
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_info = {
                'traceback': exception_traceback,
                'error': error_message
            }
            return StatusType.CRASHED, self.cost_for_crash, 0.0, additional_info

        if obj.exit_status in (pynisher.TimeoutException, pynisher.MemorylimitException):
            # Even if the pynisher thinks that a timeout or memout occured,
            # it can be that the target algorithm wrote something into the queue
            #  - then we treat it as a succesful run
            try:
                info = autosklearn.evaluation.util.read_queue(queue)
                result = info[-1]['loss']
                status = info[-1]['status']
                additional_run_info = info[-1]['additional_run_info']

                if obj.stdout:
                    additional_run_info['subprocess_stdout'] = obj.stdout
                if obj.stderr:
                    additional_run_info['subprocess_stderr'] = obj.stderr

                if obj.exit_status is pynisher.TimeoutException:
                    additional_run_info['info'] = 'Run stopped because of timeout.'
                elif obj.exit_status is pynisher.MemorylimitException:
                    additional_run_info['info'] = 'Run stopped because of memout.'

                if status in [StatusType.SUCCESS, StatusType.DONOTADVANCE]:
                    cost = result
                else:
                    cost = self.worst_possible_result

            except Empty:
                info = None
                if obj.exit_status is pynisher.TimeoutException:
                    status = StatusType.TIMEOUT
                    additional_run_info = {'error': 'Timeout'}
                elif obj.exit_status is pynisher.MemorylimitException:
                    status = StatusType.MEMOUT
                    additional_run_info = {
                        'error': 'Memout (used more than %d MB).' % self.memory_limit
                    }
                else:
                    raise ValueError(obj.exit_status)
                cost = self.worst_possible_result

        elif obj.exit_status is TAEAbortException:
            info = None
            status = StatusType.ABORT
            cost = self.worst_possible_result
            additional_run_info = {'error': 'Your configuration of '
                                            'auto-sklearn does not work!',
                                   'exit_status': _encode_exit_status(obj.exit_status),
                                   'subprocess_stdout': obj.stdout,
                                   'subprocess_stderr': obj.stderr,
                                   }

        else:
            try:
                info = autosklearn.evaluation.util.read_queue(queue)
                result = info[-1]['loss']
                status = info[-1]['status']
                additional_run_info = info[-1]['additional_run_info']

                if obj.exit_status == 0:
                    cost = result
                else:
                    status = StatusType.CRASHED
                    cost = self.worst_possible_result
                    additional_run_info['info'] = 'Run treated as crashed ' \
                                                  'because the pynisher exit ' \
                                                  'status %s is unknown.' % \
                                                  str(obj.exit_status)
                    additional_run_info['exit_status'] = _encode_exit_status(obj.exit_status)
                    additional_run_info['subprocess_stdout'] = obj.stdout
                    additional_run_info['subprocess_stderr'] = obj.stderr
            except Empty:
                info = None
                additional_run_info = {
                    'error': 'Result queue is empty',
                    'exit_status': _encode_exit_status(obj.exit_status),
                    'subprocess_stdout': obj.stdout,
                    'subprocess_stderr': obj.stderr,
                    'exitcode': obj.exitcode
                }
                status = StatusType.CRASHED
                cost = self.worst_possible_result

        if (
            (self.budget_type is None or budget == 0)
            and status == StatusType.DONOTADVANCE
        ):
            status = StatusType.SUCCESS

        if not isinstance(additional_run_info, dict):
            additional_run_info = {'message': additional_run_info}

        if (
            info is not None
            and self.resampling_strategy in ('holdout-iterative-fit', 'cv-iterative-fit')
            and status != StatusType.CRASHED
        ):
            learning_curve = autosklearn.evaluation.util.extract_learning_curve(info)
            learning_curve_runtime = autosklearn.evaluation.util.extract_learning_curve(
                info, 'duration'
            )
            if len(learning_curve) > 1:
                additional_run_info['learning_curve'] = learning_curve
                additional_run_info['learning_curve_runtime'] = learning_curve_runtime

            train_learning_curve = autosklearn.evaluation.util.extract_learning_curve(
                info, 'train_loss'
            )
            if len(train_learning_curve) > 1:
                additional_run_info['train_learning_curve'] = train_learning_curve
                additional_run_info['learning_curve_runtime'] = learning_curve_runtime

            if self._get_validation_loss:
                validation_learning_curve = autosklearn.evaluation.util.extract_learning_curve(
                    info, 'validation_loss',
                )
                if len(validation_learning_curve) > 1:
                    additional_run_info['validation_learning_curve'] = \
                        validation_learning_curve
                    additional_run_info[
                        'learning_curve_runtime'] = learning_curve_runtime

            if self._get_test_loss:
                test_learning_curve = autosklearn.evaluation.util.extract_learning_curve(
                    info, 'test_loss',
                )
                if len(test_learning_curve) > 1:
                    additional_run_info['test_learning_curve'] = test_learning_curve
                    additional_run_info[
                        'learning_curve_runtime'] = learning_curve_runtime

        if isinstance(config, int):
            origin = 'DUMMY'
        else:
            origin = getattr(config, 'origin', 'UNKNOWN')
        additional_run_info['configuration_origin'] = origin

        runtime = float(obj.wall_clock_time)

        autosklearn.evaluation.util.empty_queue(queue)
        self.logger.debug(
            'Finished function evaluation. Status: %s, Cost: %f, Runtime: %f, Additional %s',
            status, cost, runtime, additional_run_info,
        )
        return status, cost, runtime, additional_run_info
