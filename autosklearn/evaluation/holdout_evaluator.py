# -*- encoding: utf-8 -*-
from __future__ import print_function
import signal

import numpy as np
from smac.tae.execute_ta_run import StatusType

from autosklearn.constants import *
from autosklearn.evaluation.resampling import split_data
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator, \
    _get_base_dict


__all__ = [
    'HoldoutEvaluator',
    'eval_holdout',
    'eval_holdout_on_subset',
    'eval_iterative_holdout',
    'eval_iterative_holdout_on_subset'
]


class HoldoutEvaluator(AbstractEvaluator):

    def __init__(self, datamanager, output_dir,
                 configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 num_run=None):
        super(HoldoutEvaluator, self).__init__(
            datamanager, output_dir, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run)

        classification = datamanager.info['task'] in CLASSIFICATION_TASKS
        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
            split_data(datamanager.data['X_train'],
                       datamanager.data['Y_train'],
                       classification=classification)

    def fit_predict_and_loss(self):
        self.model.fit(self.X_train, self.Y_train)
        return self.predict_and_loss()

    def iterative_fit(self):
        Xt, fit_params = self.model.pre_transform(self.X_train, self.Y_train)
        if not self.model.estimator_supports_iterative_fit():
            print("Model does not support iterative_fit(), reverting to " \
                "regular fit().")

            self.model.fit_estimator(Xt, self.Y_train, **fit_params)
            return

        n_iter = 1
        while not self.model.configuration_fully_fitted():
            self.model.iterative_fit(Xt, self.Y_train, n_iter=n_iter,
                                     **fit_params)
            loss, Y_optimization_pred, Y_valid_pred, Y_test_pred \
                = self.predict_and_loss()
            self.file_output(loss, Y_optimization_pred,
                             Y_valid_pred, Y_test_pred)
            n_iter += 2

    def _predict(self):
        Y_optimization_pred = self.predict_function(self.X_optimization,
                                                    self.model, self.task_type,
                                                    self.Y_train)
        if self.X_valid is not None:
            Y_valid_pred = self.predict_function(self.X_valid, self.model,
                                                 self.task_type,
                                                 self.Y_train)
        else:
            Y_valid_pred = None
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model,
                                                self.task_type,
                                                self.Y_train)
        else:
            Y_test_pred = None

        return Y_optimization_pred, Y_valid_pred, Y_test_pred

    def predict_and_loss(self):
        Y_optimization_pred, Y_valid_pred, Y_test_pred = self._predict()
        loss = self._loss(self.Y_optimization, Y_optimization_pred)
        return loss, Y_optimization_pred, Y_valid_pred, Y_test_pred


# create closure for evaluating an algorithm
def eval_holdout(queue, configuration, data, tmp_dir, seed, num_run,
                 iterative=False):
    evaluator = HoldoutEvaluator(data, tmp_dir, configuration,
                                 seed=seed,
                                 num_run=num_run,
                                 **_get_base_dict())

    def signal_handler(signum, frame):
        print('Received signal %s. Aborting Training!', str(signum))
        global evaluator
        duration, result, seed, run_info = evaluator.finish_up()
        # TODO use status type for stopped, but yielded a result
        queue.put((duration, result, seed, run_info, StatusType.SUCCESS))

    def empty_signal_handler(signum, frame):
        pass

    if iterative:
        signal.signal(15, signal_handler)
        evaluator.iterative_fit()
        signal.signal(15, empty_signal_handler)
        duration, result, seed, run_info = evaluator.finish_up()
    else:
        loss, opt_pred, valid_pred, test_pred = evaluator.fit_predict_and_loss()
        duration, result, seed, run_info = evaluator.finish_up(
            loss, opt_pred, valid_pred, test_pred)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))


def eval_iterative_holdout(queue, configuration, data, tmp_dir, seed, num_run):
    eval_holdout(queue, configuration, data, tmp_dir, seed, num_run, True)


def eval_holdout_on_subset(queue, configuration, n_data_subsample, data,
                           tmp_dir, seed, num_run, iterative=False):
    # Get full optimization split - TODO refactor this!
    evaluator_ = HoldoutEvaluator(data, tmp_dir, configuration,
                                  seed=seed,
                                  num_run=num_run,
                                  **_get_base_dict())
    X_optimization = evaluator_.X_optimization
    Y_optimization = evaluator_.Y_optimization
    del evaluator_

    n_data = data.data['X_train'].shape[0]
    # TODO get random states
    # get pointers to the full data
    Xfull = data.data['X_train']
    Yfull = data.data['Y_train']
    # create a random subset
    indices = np.random.randint(0, n_data, n_data_subsample)
    data.data['X_train'] = Xfull[indices, :]
    data.data['Y_train'] = Yfull[indices]

    evaluator = HoldoutEvaluator(data, tmp_dir, configuration,
                                 seed=seed,
                                 num_run=num_run,
                                 **_get_base_dict())

    def signal_handler(signum, frame):
        print('Received signal %s. Aborting Training!', str(signum))
        global evaluator
        duration, result, seed, run_info = evaluator.finish_up()
        # TODO use status type for stopped, but yielded a result
        queue.put((duration, result, seed, run_info, StatusType.SUCCESS))

    def empty_signal_handler(signum, frame):
        pass

    if iterative:
        signal.signal(15, signal_handler)
        evaluator.iterative_fit()
        signal.signal(15, empty_signal_handler)
        loss, _opt_pred, valid_pred, test_pred = evaluator.predict_and_loss()
    else:
        loss, _opt_pred, valid_pred, test_pred = evaluator.fit_predict_and_loss()

    # predict on the whole dataset, needed for ensemble
    opt_pred = evaluator.predict_function(X_optimization, evaluator.model,
                                          evaluator.task_type, Yfull)
    # TODO remove this hack
    evaluator.output_y_test = False
    evaluator.Y_optimization = Y_optimization

    duration, result, seed, run_info = evaluator.finish_up(
        loss, opt_pred, valid_pred, test_pred)
    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))


def eval_iterative_holdout_on_subset(queue, configuration, n_data_subsample,
                                     data, tmp_dir, seed, num_run):
    eval_holdout_on_subset(queue, configuration, n_data_subsample, data,
                           tmp_dir, seed, num_run, True)