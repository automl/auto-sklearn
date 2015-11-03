# -*- encoding: utf-8 -*-
from __future__ import print_function
import abc
import os
import time
import traceback

import numpy as np
import lockfile
from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from autosklearn.constants import *
from autosklearn.evaluation.util import get_new_run_num
from autosklearn.util import Backend


__all__ = [
    'AbstractEvaluator'
]


class MyDummyClassifier(DummyClassifier):
    def __init__(self, configuration, random_states):
        super(MyDummyClassifier, self).__init__(strategy="most_frequent")

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyClassifier, self).fit(np.ones((X.shape[0], 1)), y,
                                                  sample_weight=sample_weight)

    def predict_proba(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyClassifier, self).predict_proba(new_X)


class MyDummyRegressor(DummyRegressor):
    def __init__(self, configuration, random_states):
        super(MyDummyRegressor, self).__init__(strategy='mean')

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyRegressor, self).fit(np.ones((X.shape[0], 1)), y,
                                                 sample_weight=sample_weight)

    def predict(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X)


class AbstractEvaluator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, Datamanager, configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_dir=None,
                 output_y_test=False,
                 num_run=None):

        self.starttime = time.time()

        self.configuration = configuration
        self.D = Datamanager

        self.X_valid = Datamanager.data.get('X_valid')
        self.X_test = Datamanager.data.get('X_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task']
        self.seed = seed

        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir

        self.output_y_test = output_y_test
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type in REGRESSION_TASKS:
            if self.configuration is None:
                self.model_class = MyDummyRegressor
            else:
                self.model_class = ParamSklearnRegressor
            self.predict_function = self.predict_regression
        else:
            if self.configuration is None:
                self.model_class = MyDummyClassifier
            else:
                self.model_class = ParamSklearnClassifier
            self.predict_function = self.predict_proba

        if num_run is None:
            num_run = get_new_run_num()
        self.num_run = num_run

        self.backend = Backend(None, self.output_dir)

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    # This function does everything necessary after the fitting is done:
    #        predicting
    #        saving the files for the ensembles_statistics
    #        generate output for SMAC
    # We use it as the signal handler so we can recycle the code for the
    # normal usecase and when the runsolver kills us here :)
    def finish_up(self):
        try:
            self.duration = time.time() - self.starttime
            result, additional_run_info = self.file_output()
            print('Result for ParamILS: %s, %f, 1, %f, %d, %s' %
                  ('SAT', abs(self.duration), result, self.seed,
                   additional_run_info))
        except Exception as e:
            self.duration = time.time() - self.starttime

            print(traceback.format_exc())
            print('Result for ParamILS: %s, %f, 1, %f, %d, %s' %
                  ('TIMEOUT', abs(self.duration), 1.0, self.seed,
                   'No results were produced! Error is %s' % str(e)))

    def file_output(self):
        seed = os.environ.get('AUTOSKLEARN_SEED')
        errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = self.predict()
        num_run = str(self.num_run).zfill(5)

        if self.output_y_test:
            try:
                os.makedirs(self.output_dir)
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

        self.duration = time.time() - self.starttime
        err = errs[self.D.info['metric']]
        additional_run_info = ';'.join(['%s: %s' %
            (METRIC_TO_STRING[metric] if metric in METRIC_TO_STRING else metric,
                                                                     value)
                                        for metric, value in errs.items()])
        additional_run_info += ';' + 'duration: ' + str(self.duration)
        additional_run_info += ';' + 'num_run:' + num_run
        return err, additional_run_info

    def predict_proba(self, X, model, task_type, Y_train=None):
        Y_pred = model.predict_proba(X, batch_size=1000)

        if task_type == MULTILABEL_CLASSIFICATION:
            Y_pred = np.hstack([Y_pred[i][:, -1].reshape((-1, 1))
                                for i in range(len(Y_pred))])

        elif task_type == BINARY_CLASSIFICATION:
            if len(Y_pred.shape) != 1:
                Y_pred = Y_pred[:, 1].reshape(-1, 1)

        elif task_type == MULTICLASS_CLASSIFICATION:
            pass

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def predict_regression(self, X, model, task_type, Y_train=None):
        Y_pred = model.predict(X)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction, Y_train):
        num_classes = self.D.info['label_num']

        if self.task_type == MULTICLASS_CLASSIFICATION and \
                prediction.shape[1] < num_classes:
            classes = list(np.unique(self.D.data['Y_train']))
            if num_classes == prediction.shape[1]:
                return prediction

            if Y_train is not None:
                classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes))
            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
