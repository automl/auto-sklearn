# -*- encoding: utf-8 -*-
from __future__ import print_function
import signal

import numpy as np
from smac.tae.execute_ta_run import StatusType

from autosklearn.constants import *
from autosklearn.evaluation.resampling import split_data
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator


__all__ = [
    'HoldoutEvaluator',
    'eval_holdout',
    'eval_iterative_holdout',
]


class HoldoutEvaluator(AbstractEvaluator):

    def __init__(self, datamanager, backend,
                 configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 num_run=None,
                 subsample=None):
        super(HoldoutEvaluator, self).__init__(
            datamanager, backend, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run,
            subsample=subsample)

        classification = datamanager.info['task'] in CLASSIFICATION_TASKS
        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
            split_data(datamanager.data['X_train'],
                       datamanager.data['Y_train'],
                       classification=classification)

    def fit_predict_and_loss(self):
        num_train_points = self.X_train.shape[0]

        if self.subsample is not None:
            n_data_subsample = min(self.subsample, num_train_points)
            indices = np.array(([True] * n_data_subsample) + \
                               ([False] * (num_train_points - n_data_subsample)),
                               dtype=np.bool)
            rs = np.random.RandomState(self.seed)
            rs.shuffle(indices)
            X_train, Y_train = self.X_train[indices], self.Y_train[indices]
        else:
            X_train, Y_train = self.X_train, self.Y_train

        self._fit_and_suppress_warnings(self.model, X_train, Y_train)
        return self.predict_and_loss()

    def iterative_fit(self):
        num_train_points = self.X_train.shape[0]

        if self.subsample is not None:
            n_data_subsample = min(self.subsample, num_train_points)
            indices = np.array(([True] * n_data_subsample) + \
                               (
                               [False] * (num_train_points - n_data_subsample)),
                               dtype=np.bool)
            rs = np.random.RandomState(self.seed)
            rs.shuffle(indices)
            X_train, Y_train = self.X_train[indices], self.Y_train[indices]
        else:
            X_train, Y_train = self.X_train, self.Y_train

        Xt, fit_params = self.model.pre_transform(X_train, Y_train)
        if not self.model.estimator_supports_iterative_fit():
            print("Model does not support iterative_fit(), reverting to " \
                "regular fit().")

            self.model.fit_estimator(Xt, Y_train, **fit_params)
            return

        n_iter = 1
        while not self.model.configuration_fully_fitted():
            self.model.iterative_fit(Xt, Y_train, n_iter=n_iter,
                                     **fit_params)
            loss, Y_optimization_pred, Y_valid_pred, Y_test_pred \
                = self.predict_and_loss()
            self.file_output(loss, Y_optimization_pred,
                             Y_valid_pred, Y_test_pred)
            n_iter *= 2

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
def eval_holdout(queue, config, data, backend, seed, num_run,
                 subsample, with_predictions, all_scoring_functions,
                 output_y_test, iterative=False):
    evaluator = HoldoutEvaluator(data, backend, config,
                                 seed=seed,
                                 num_run=num_run,
                                 subsample=subsample,
                                 with_predictions=with_predictions,
                                 all_scoring_functions=all_scoring_functions,
                                 output_y_test=output_y_test)

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


def eval_iterative_holdout(queue, config, data, backend, seed,
                           num_run, subsample, with_predictions,
                           all_scoring_functions, output_y_test):
    eval_holdout(queue, config, data, backend, seed, num_run, subsample,
                 with_predictions, all_scoring_functions, output_y_test,
                 iterative = True)
