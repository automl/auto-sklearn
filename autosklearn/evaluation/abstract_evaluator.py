import os
import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from smac.tae.execute_ta_run import StatusType

import autosklearn.pipeline.classification
import autosklearn.pipeline.regression
from autosklearn.constants import *
from autosklearn.pipeline.implementations.util import convert_multioutput_multiclass_to_multilabel
from autosklearn.metrics import calculate_score
from autosklearn.util.logging_ import get_logger

from ConfigSpace import Configuration


__all__ = [
    'AbstractEvaluator'
]


class MyDummyClassifier(DummyClassifier):
    def __init__(self, configuration, random_state, init_params=None):
        self.configuration = configuration
        if configuration == 1:
            super(MyDummyClassifier, self).__init__(strategy="uniform")
        else:
            super(MyDummyClassifier, self).__init__(strategy="most_frequent")

    def pre_transform(self, X, y, fit_params=None):
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
    def __init__(self, configuration, random_state, init_params=None):
        self.configuration = configuration
        if configuration == 1:
            super(MyDummyRegressor, self).__init__(strategy='mean')
        else:
            super(MyDummyRegressor, self).__init__(strategy='median')

    def pre_transform(self, X, y, fit_params=None):
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
    def __init__(self, datamanager, backend, queue, metric,
                 configuration=None,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_hat_optimization=True,
                 num_run=None,
                 subsample=None,
                 include=None,
                 exclude=None,
                 disable_file_output=False):

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.queue = queue

        self.datamanager = datamanager
        self.include = include
        self.exclude = exclude

        self.X_valid = datamanager.data.get('X_valid')
        self.X_test = datamanager.data.get('X_test')

        self.metric = metric
        self.task_type = datamanager.info['task']
        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization
        self.all_scoring_functions = all_scoring_functions
        self.disable_file_output = disable_file_output

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

        categorical_mask = []
        for feat in datamanager.feat_type:
            if feat.lower() == 'numerical':
                categorical_mask.append(False)
            elif feat.lower() == 'categorical':
                categorical_mask.append(True)
            else:
                raise ValueError(feat)
        if np.sum(categorical_mask) > 0:
            self._init_params = {'one_hot_encoding:categorical_features':
                                     categorical_mask}
        else:
            self._init_params = {}

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        self.subsample = subsample

        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.datamanager.name)
        self.logger = get_logger(logger_name)

    def _get_model(self):
        if not isinstance(self.configuration, Configuration):
            model = self.model_class(configuration=self.configuration,
                                     random_state=self.seed,
                                     init_params=self._init_params)
        else:
            dataset_properties = {'task': self.task_type,
                                  'sparse': self.datamanager.info['is_sparse'] == 1,
                                  'multilabel': self.task_type ==
                                                MULTILABEL_CLASSIFICATION,
                                  'multiclass': self.task_type ==
                                                MULTICLASS_CLASSIFICATION}
            model = self.model_class(config=self.configuration,
                                     dataset_properties=dataset_properties,
                                     random_state=self.seed,
                                     include=self.include,
                                     exclude=self.exclude,
                                     init_params=self._init_params)
        return model

    def _loss(self, y_true, y_hat):
        if not isinstance(self.configuration, Configuration):
            if self.all_scoring_functions:
                return {self.metric: 1.0}
            else:
                return 1.0

        score = calculate_score(
            y_true, y_hat, self.task_type, self.metric,
            all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, '__len__'):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        return err

    def finish_up(self, loss, opt_pred, valid_pred, test_pred,
                  file_output=True, final_call=True):
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred)
        else:
            loss_, additional_run_info_ = None, {}

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        num_run = str(self.num_run).zfill(5)
        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = {metric_name: value for metric_name, value in
                               loss_.items()}
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run

        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': StatusType.SUCCESS}
        if final_call:
            rval_dict['final_queue_element'] = True

        self.queue.put(rval_dict)

    def file_output(self, Y_optimization_pred, Y_valid_pred, Y_test_pred):
        if self.disable_file_output is True:
            return None, {}

        seed = self.seed

        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is not None and \
                self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return 1.0, {'error': "Targets %s and prediction %s don't have "
                                  "the same length. Probably training didn't "
                                  "finish" % (self.Y_optimization.shape,
                                              Y_optimization_pred.shape)}

        if not np.all(np.isfinite(Y_optimization_pred)):
            return 1.0, {'error': 'Model predictions for optimization set ' \
                                  'contains NaNs.'}
        for y, s in [[Y_valid_pred, 'validation'], [Y_test_pred, 'test']]:
            if y is not None and not np.all(np.isfinite(y)):
                return 1.0, {'error': 'Model predictions for %s set contains '
                                      'NaNs.' % s}

        num_run = str(self.num_run).zfill(5)

        if not isinstance(self.disable_file_output, list) or \
                'model' not in self.disable_file_output:
            if os.path.exists(self.backend.get_model_dir()):
                self.backend.save_model(self.model, self.num_run, seed)

        if not isinstance(self.disable_file_output, list) or \
                'y_optimization' not in self.disable_file_output:
            if self.output_y_hat_optimization:
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
                                 file=None, line=None):
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
                                 file=None, line=None):
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
        num_classes = self.datamanager.info['label_num']

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
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            model.fit(X, y)

        return model

