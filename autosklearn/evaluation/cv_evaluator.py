# -*- encoding: utf-8 -*-
import signal

import numpy as np
from smac.tae.execute_ta_run import StatusType

from autosklearn.evaluation.resampling import get_CV_fold
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator


__all__ = [
    'CVEvaluator',
    'eval_cv',
    'eval_partial_cv',
]


class CVEvaluator(AbstractEvaluator):
    def __init__(self, Datamanager, backend,
                 configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 cv_folds=10,
                 num_run=None,
                 subsample=None,
                 keep_models=False):
        super(CVEvaluator, self).__init__(
            Datamanager, backend, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run,
            subsample=subsample)

        self.cv_folds = cv_folds
        self.X_train = self.D.data['X_train']
        self.Y_train = self.D.data['Y_train']
        self.Y_optimization = None
        self.Y_targets = [None] * cv_folds
        self.models = [None] * cv_folds
        self.indices = [None] * cv_folds

        # Necessary for full CV. Makes full CV not write predictions if only
        # a subset of folds is evaluated but time is up. Complicated, because
        #  code must also work for partial CV, where we want exactly the
        # opposite.
        self.partial = True
        self.keep_models = keep_models

    def fit_predict_and_loss(self):
        Y_optimization_pred = [None] * self.cv_folds
        Y_valid_pred = [None] * self.cv_folds
        Y_test_pred = [None] * self.cv_folds

        self.partial = False
        for fold in range(self.cv_folds):
            opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(
                fold)

            Y_optimization_pred[fold] = opt_pred
            Y_valid_pred[fold] = valid_pred
            Y_test_pred[fold] = test_pred

        Y_targets = self.Y_targets

        Y_optimization_pred = np.concatenate(
            [Y_optimization_pred[i] for i in range(self.cv_folds)
             if Y_optimization_pred[i] is not None])
        Y_targets = np.concatenate([Y_targets[i] for i in range(self.cv_folds)
                                    if Y_targets[i] is not None])

        if self.X_valid is not None:
            Y_valid_pred = np.array([Y_valid_pred[i]
                                     for i in range(self.cv_folds)
                                     if Y_valid_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_valid_pred.shape) == 3:
                Y_valid_pred = np.nanmean(Y_valid_pred, axis=0)
        else:
            Y_valid_pred = None

        if self.X_test is not None:
            Y_test_pred = np.array([Y_test_pred[i]
                                    for i in range(self.cv_folds)
                                    if Y_test_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_test_pred.shape) == 3:
                Y_test_pred = np.nanmean(Y_test_pred, axis=0)
        else:
            Y_test_pred = None

        self.Y_optimization = Y_targets
        loss = self._loss(Y_targets, Y_optimization_pred)
        return loss, Y_optimization_pred, Y_valid_pred, Y_test_pred

    def partial_fit_predict_and_loss(self, fold):
        opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(fold)
        loss = self._loss(self.Y_targets[fold], opt_pred)
        return loss, opt_pred, valid_pred, test_pred

    def _partial_fit_and_predict(self, fold):
        model = self.model_class(self.configuration, self.seed)

        train_indices, test_indices = self.get_train_test_split(fold)

        if self.subsample is not None:
            n_data_subsample = min(self.subsample, len(train_indices))
            indices = np.array(([True] * n_data_subsample) + \
                               ([False] * (len(train_indices) - n_data_subsample)),
                               dtype=np.bool)
            rs = np.random.RandomState(self.seed)
            rs.shuffle(indices)
            train_indices = train_indices[indices]

        self.indices[fold] = ((train_indices, test_indices))
        self._fit_and_suppress_warnings(model,
                                        self.X_train[train_indices],
                                        self.Y_train[train_indices])

        if self.keep_models:
            self.models[fold] = model

        train_indices, test_indices = self.indices[fold]
        self.Y_targets[fold] = self.Y_train[test_indices]

        opt_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type,
                                         self.Y_train[train_indices])

        if self.X_valid is not None:
            X_valid = self.X_valid.copy()
            valid_pred = self.predict_function(X_valid, model,
                                               self.task_type,
                                               self.Y_train[train_indices])
        else:
            valid_pred = None

        if self.X_test is not None:
            X_test = self.X_test.copy()
            test_pred = self.predict_function(X_test, model,
                                              self.task_type,
                                              self.Y_train[train_indices])
        else:
            test_pred = None

        return opt_pred, valid_pred, test_pred

    def get_train_test_split(self, fold):
        return get_CV_fold(self.X_train, self.Y_train, fold=fold,
                            folds=self.cv_folds, shuffle=True,
                            random_state=self.seed)


def eval_partial_cv(queue, config, data, backend, seed, num_run, fold,
                    folds, subsample, with_predictions, all_scoring_functions,
                    output_y_test):
    evaluator = CVEvaluator(data, backend, config,
                            seed=seed,
                            num_run=num_run,
                            cv_folds=folds,
                            subsample=subsample,
                            with_predictions=with_predictions,
                            all_scoring_functions=all_scoring_functions,
                            output_y_test=output_y_test)

    loss, opt_pred, valid_pred, test_pred = \
        evaluator.partial_fit_predict_and_loss(fold)
    duration, result, seed, run_info = evaluator.finish_up(
        loss, opt_pred, valid_pred, test_pred, False)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))


# create closure for evaluating an algorithm
def eval_cv(queue, config, data, backend, seed, num_run, folds,
            subsample, with_predictions, all_scoring_functions,
            output_y_test):
    evaluator = CVEvaluator(data, backend, config,
                            seed=seed,
                            num_run=num_run,
                            cv_folds=folds,
                            subsample=subsample,
                            with_predictions=with_predictions,
                            all_scoring_functions=all_scoring_functions,
                            output_y_test=output_y_test)

    loss, opt_pred, valid_pred, test_pred = evaluator.fit_predict_and_loss()
    duration, result, seed, run_info = evaluator.finish_up(
        loss, opt_pred, valid_pred, test_pred)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))
