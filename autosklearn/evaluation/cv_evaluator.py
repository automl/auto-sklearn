# -*- encoding: utf-8 -*-
import signal

import numpy as np
from smac.tae.execute_ta_run import StatusType

from ConfigSpace import Configuration
from autosklearn.evaluation.resampling import get_CV_fold
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.constants import MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION


__all__ = [
    'CVEvaluator',
    'eval_cv',
    'eval_partial_cv',
    'eval_partial_cv_iterative'
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
                 keep_models=False,
                 include=None,
                 exclude=None):
        super(CVEvaluator, self).__init__(
            Datamanager, backend, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run,
            subsample=subsample,
            include=include,
            exclude=exclude)

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
        dataset_properties = {'task': self.task_type,
                              'sparse': self.D.info['is_sparse'] == 1,
                              'is_multilabel': self.task_type ==
                                               MULTILABEL_CLASSIFICATION,
                              'is_multiclass': self.task_type ==
                                               MULTICLASS_CLASSIFICATION}
        if not isinstance(self.configuration, Configuration):
            # Dummy classifiers
            model = self.model_class(configuration=self.configuration,
                                     random_state=self.seed)
        else:
            model = self.model_class(config=self.configuration,
                                     random_state=self.seed,
                                     include=self.include,
                                     exclude=self.exclude,
                                     dataset_properties=dataset_properties)

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

    def partial_iterative_fit(self, fold):
        dataset_properties = {'task': self.task_type,
                              'sparse': self.D.info['is_sparse'] == 1,
                              'is_multilabel': self.task_type ==
                                               MULTILABEL_CLASSIFICATION,
                              'is_multiclass': self.task_type ==
                                               MULTICLASS_CLASSIFICATION}
        model = self.model_class(config=self.configuration,
                                 dataset_properties=dataset_properties,
                                 random_state=self.seed,
                                 include=self.include,
                                 exclude=self.exclude,
                                 init_params=self._init_params)
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
        X_train = self.X_train[train_indices]
        Y_train = self.Y_train[train_indices]
        self.Y_targets[fold] = self.Y_train[test_indices]

        Xt, fit_params = model.pre_transform(X_train, Y_train)
        if not model.estimator_supports_iterative_fit():
            print("Model does not support iterative_fit(), reverting to " \
                "regular fit().")

            model.fit_estimator(Xt, Y_train, **fit_params)
            self.models[fold] = model
            return

        n_iter = 1
        self.models[fold] = model
        while not model.configuration_fully_fitted():
            model.iterative_fit(Xt, Y_train, n_iter=n_iter,
                                **fit_params)
            n_iter *= 2

    def predict_and_loss(self):
        # Only called by finish_up for iterative fit
        for fold, model in enumerate(self.models):
            if model is None:
                continue

            train_indices, test_indices = self.indices[fold]

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

            loss = self._loss(self.Y_targets[fold], opt_pred)

            return loss, opt_pred, valid_pred, test_pred

    def get_train_test_split(self, fold):
        return get_CV_fold(self.X_train, self.Y_train, fold=fold,
                            folds=self.cv_folds, shuffle=True,
                            random_state=self.seed)


def eval_partial_cv(queue, config, data, backend, seed, num_run, instance,
                    folds, subsample, with_predictions, all_scoring_functions,
                    output_y_test, include, exclude, iterative=False):
    global evaluator
    evaluator = CVEvaluator(data, backend, config,
                            seed=seed,
                            num_run=num_run,
                            cv_folds=folds,
                            subsample=subsample,
                            with_predictions=with_predictions,
                            all_scoring_functions=all_scoring_functions,
                            output_y_test=False,
                            include=include,
                            exclude=exclude)


    def signal_handler(signum, frame):
        print('Received signal %s. Aborting Training!' % str(signum))
        global evaluator
        duration, result, seed, run_info = evaluator.finish_up(file_output=False)
        # TODO use status type for stopped, but yielded a result
        queue.put((duration, result, seed, run_info, StatusType.SUCCESS))

    def empty_signal_handler(signum, frame):
        pass

    if iterative:
        signal.signal(signal.SIGALRM, signal_handler)
        evaluator.partial_iterative_fit(instance)
        signal.signal(signal.SIGALRM, empty_signal_handler)
        duration, result, seed, run_info = evaluator.finish_up(file_output=False)
    else:
        loss, opt_pred, valid_pred, test_pred = \
            evaluator.partial_fit_predict_and_loss(instance)
        duration, result, seed, run_info = evaluator.finish_up(
            loss, opt_pred, valid_pred, test_pred, file_output=False)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))


def eval_partial_cv_iterative(queue, config, data, backend, seed, num_run,
                              instance, folds, subsample, with_predictions,
                              all_scoring_functions, output_y_test,
                              include, exclude):
    eval_partial_cv(queue=queue, config=config, data=data, backend=backend,
                    seed=seed, num_run=num_run, instance=instance, folds=folds,
                    subsample=subsample, with_predictions=with_predictions,
                    all_scoring_functions=all_scoring_functions,
                    output_y_test=output_y_test, include=include,
                    exclude=exclude, iterative=True)


# create closure for evaluating an algorithm
def eval_cv(queue, config, data, backend, seed, num_run, folds,
            subsample, with_predictions, all_scoring_functions,
            output_y_test, include, exclude):
    evaluator = CVEvaluator(data, backend, config,
                            seed=seed,
                            num_run=num_run,
                            cv_folds=folds,
                            subsample=subsample,
                            with_predictions=with_predictions,
                            all_scoring_functions=all_scoring_functions,
                            output_y_test=output_y_test,
                            include=include,
                            exclude=exclude)

    loss, opt_pred, valid_pred, test_pred = evaluator.fit_predict_and_loss()
    duration, result, seed, run_info = evaluator.finish_up(
        loss, opt_pred, valid_pred, test_pred)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))
