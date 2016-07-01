# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

import autosklearn.pipeline.classification
import autosklearn.pipeline.regression
from autosklearn.constants import *
from autosklearn.util import Backend
from autosklearn.pipeline.implementations.util import convert_multioutput_multiclass_to_multilabel
from autosklearn.evaluation.util import calculate_score
from autosklearn.util.logging_ import get_logger

from ConfigSpace import Configuration


__all__ = [
    'AbstractEvaluator'
]


class MyDummyClassifier(DummyClassifier):
    def __init__(self, configuration, random_states):
        self.configuration = configuration
        if configuration == 1:
            super(MyDummyClassifier, self).__init__(strategy="uniform")
        else:
            super(MyDummyClassifier, self).__init__(strategy="most_frequent")

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyClassifier, self).fit(np.ones((X.shape[0], 1)), y,
                                                  sample_weight=sample_weight)

    def fit_estimator(self, X, y, fit_params=None):
        return self.fit(X, y)

    def predict_proba(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        probas = super(MyDummyClassifier, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def estimator_supports_iterative_fit(self):
        return False


class MyDummyRegressor(DummyRegressor):
    def __init__(self, configuration, random_states):
        self.configuration = configuration
        if configuration == 1:
            super(MyDummyRegressor, self).__init__(strategy='mean')
        else:
            super(MyDummyRegressor, self).__init__(strategy='median')

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyRegressor, self).fit(np.ones((X.shape[0], 1)), y,
                                                 sample_weight=sample_weight)

    def fit_estimator(self, X, y, fit_params=None):
        return self.fit(X, y)

    def predict(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self):
        return False


class AbstractEvaluator(object):
    def __init__(self, Datamanager, backend, configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 num_run=None,
                 subsample=None,):

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend

        self.D = Datamanager

        self.X_valid = Datamanager.data.get('X_valid')
        self.X_test = Datamanager.data.get('X_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task']
        self.seed = seed

        self.output_y_test = output_y_test
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type in REGRESSION_TASKS:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyRegressor
            else:
                self.model_class = \
                    autosklearn.pipeline.regression.SimpleRegressionPipeline
            self.predict_function = self._predict_regression
        else:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyClassifier
            else:
                self.model_class = \
                    autosklearn.pipeline.classification.SimpleClassificationPipeline
            self.predict_function = self._predict_proba

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        self.subsample = subsample

        self.model = self.model_class(self.configuration, self.seed)

        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.D.name)
        self.logger = get_logger(logger_name)

    def fit_predict_and_loss(self):
        """Fit model(s) according to resampling strategy, predict for the
        validation set and return the loss and predictions on the validation
        set.

        Provides a closed interface in which all steps of the target
        algorithm are performed without any communication with other
        processes. Useful for cross-validation because it allows to train a
        model, predict for the validation set and then forget the model in
        order to save main memory.
        """
        raise NotImplementedError()

    def iterative_fit(self):
        """Fit a model iteratively.

        Fitting can be interrupted in order to use a partially trained model."""
        raise NotImplementedError()

    def predict_and_loss(self):
        """Use current model to predict on the validation set and calculate
        loss.

         Should be used when using iterative fitting."""
        raise NotImplementedError()

    def predict(self):
        """Use the current model to predict on the validation set.

        Should only be used to create dummy predictions."""
        raise NotImplementedError()

    def _loss(self, y_true, y_hat):
        if not isinstance(self.configuration, Configuration):
            if self.all_scoring_functions:
                return {self.metric: 1.0}
            else:
                return 1.0

        score = calculate_score(
            y_true, y_hat, self.task_type,
            self.metric, self.D.info['label_num'],
            all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, '__len__'):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        return err

    def finish_up(self, loss=None, opt_pred=None, valid_pred=None,
                  test_pred=None, file_output=True):
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        # try:
        self.duration = time.time() - self.starttime
        if loss is None:
            loss, opt_pred, valid_pred, test_pred = self.predict_and_loss()

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                loss, opt_pred, valid_pred, test_pred)
        else:
            loss_, additional_run_info_ = None, None

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        num_run = str(self.num_run).zfill(5)
        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.D.info['metric']]
        else:
            loss_ = {}
        additional_run_info = ';'.join(['%s: %s' %
                                (METRIC_TO_STRING[
                                     metric] if metric in METRIC_TO_STRING else metric,
                                 value)
                                for metric, value in loss_.items()])
        additional_run_info += ';' + 'duration: ' + str(self.duration)
        additional_run_info += ';' + 'num_run:' + num_run

        return self.duration, loss, self.seed, additional_run_info

    def file_output(self, loss, Y_optimization_pred, Y_valid_pred, Y_test_pred):
        seed = self.seed

        if self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return 2.0, "Targets %s and prediction %s don't have the same " \
            "length. Probably training didn't finish" % (
                self.Y_optimization.shape, Y_optimization_pred.shape)

        if not np.all(np.isfinite(Y_optimization_pred)):
            return 2.0, 'Model predictions for optimization set contains NaNs.'
        if Y_valid_pred is not None and \
                not np.all(np.isfinite(Y_valid_pred)):
            return 2.0, 'Model predictions for validation set contains NaNs.'
        if Y_test_pred is not None and \
                not np.all(np.isfinite(Y_test_pred)):
            return 2.0, 'Model predictions for test set contains NaNs.'

        num_run = str(self.num_run).zfill(5)
        if os.path.exists(self.backend.get_model_dir()):
            self.backend.save_model(self.model, self.num_run, seed)

        if self.output_y_test:
            try:
                os.makedirs(self.backend.output_directory)
            except OSError:
                pass
            self.backend.save_targets_ensemble(self.Y_optimization)

        self.backend.save_predictions_as_npy(Y_optimization_pred, 'ensemble',
                                             seed, num_run)

        if Y_valid_pred is not None:
            self.backend.save_predictions_as_npy(Y_valid_pred, 'valid',
                                                 seed, num_run)

        if Y_test_pred is not None:
            self.backend.save_predictions_as_npy(Y_test_pred, 'test',
                                                 seed, num_run)

        return None, None

    def _predict_proba(self, X, model, task_type, Y_train):
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict_proba(X, batch_size=1000)

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X, model, task_type, Y_train=None):
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict(X)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction, Y_train):
        num_classes = self.D.info['label_num']

        if self.task_type == MULTICLASS_CLASSIFICATION and \
                prediction.shape[1] < num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction

    def _fit_and_suppress_warnings(self, model, X, y):
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None):
            self.logger.debug('%s:%s: %s:%s' %
                (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            model = model.fit(X, y)

        return model

