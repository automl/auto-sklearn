import logging
import multiprocessing
import time
import warnings
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, Union, cast

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.tae import StatusType

import autosklearn.pipeline.classification
import autosklearn.pipeline.regression
from autosklearn.constants import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    MULTILABEL_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION
)
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel
)
from autosklearn.metrics import calculate_loss, Scorer
from autosklearn.util.backend import Backend
from autosklearn.util.logging_ import PicklableClientLogger, get_named_client_logger

from ConfigSpace import Configuration


__all__ = [
    'AbstractEvaluator'
]


# General TYPE definitions for numpy
TYPE_ADDITIONAL_INFO = Dict[str, Union[int, float, str, Dict, List, Tuple]]


class MyDummyClassifier(DummyClassifier):
    def __init__(
        self,
        config: Configuration,
        random_state: np.random.RandomState,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super(MyDummyClassifier, self).__init__(strategy="uniform")
        else:
            super(MyDummyClassifier, self).__init__(strategy="most_frequent")
        self.random_state = random_state
        self.init_params = init_params

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=R0201
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[Union[np.ndarray, List]] = None
            ) -> DummyClassifier:
        return super(MyDummyClassifier, self).fit(np.ones((X.shape[0], 1)), y,
                                                  sample_weight=sample_weight)

    def fit_estimator(self, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict[str, Any]] = None) -> DummyClassifier:
        return self.fit(X, y)

    def predict_proba(self, X: np.ndarray, batch_size: int = 1000
                      ) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        probas = super(MyDummyClassifier, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:  # pylint: disable=R0201
        return None


class MyDummyRegressor(DummyRegressor):
    def __init__(
        self,
        config: Configuration,
        random_state: np.random.RandomState,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super(MyDummyRegressor, self).__init__(strategy='mean')
        else:
            super(MyDummyRegressor, self).__init__(strategy='median')
        self.random_state = random_state
        self.init_params = init_params

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=R0201
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[Union[np.ndarray, List]] = None
            ) -> DummyRegressor:
        return super(MyDummyRegressor, self).fit(np.ones((X.shape[0], 1)), y,
                                                 sample_weight=sample_weight)

    def fit_estimator(self, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict[str, Any]] = None) -> DummyRegressor:
        return self.fit(X, y)

    def predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:  # pylint: disable=R0201
        return None


def _fit_and_suppress_warnings(
    logger: Union[logging.Logger, PicklableClientLogger],
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray
) -> BaseEstimator:
    def send_warnings_to_log(
        message: Union[Warning, str],
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: Optional[TextIO] = None,
        line: Optional[str] = None,
    ) -> None:
        logger.debug('%s:%s: %s:%s' %
                     (filename, lineno, str(category), message))
        return

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        model.fit(X, y)

    return model


class AbstractEvaluator(object):
    def __init__(
        self,
        backend: Backend,
        queue: multiprocessing.Queue,
        metric: Scorer,
        port: Optional[int],
        configuration: Optional[Union[int, Configuration]] = None,
        scoring_functions: Optional[List[Scorer]] = None,
        seed: int = 1,
        output_y_hat_optimization: bool = True,
        num_run: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        disable_file_output: Union[bool, List[str]] = False,
        init_params: Optional[Dict[str, Any]] = None,
        budget: Optional[float] = None,
        budget_type: Optional[str] = None,
    ):

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.port = port
        self.queue = queue

        self.datamanager = self.backend.load_datamanager()
        self.include = include
        self.exclude = exclude

        self.X_valid = self.datamanager.data.get('X_valid')
        self.y_valid = self.datamanager.data.get('Y_valid')
        self.X_test = self.datamanager.data.get('X_test')
        self.y_test = self.datamanager.data.get('Y_test')

        self.metric = metric
        self.task_type = self.datamanager.info['task']
        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization
        self.scoring_functions = scoring_functions

        if isinstance(disable_file_output, (bool, list)):
            self.disable_file_output: Union[bool, List[str]] = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

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
                self.model_class = autosklearn.pipeline.classification.SimpleClassificationPipeline
            self.predict_function = self._predict_proba

        categorical_mask = []
        for feat in self.datamanager.feat_type:
            if feat.lower() == 'numerical':
                categorical_mask.append(False)
            elif feat.lower() == 'categorical':
                categorical_mask.append(True)
            else:
                raise ValueError(feat)
        if np.sum(categorical_mask) > 0:
            self._init_params = {
                'data_preprocessing:categorical_features':
                    categorical_mask
            }
        else:
            self._init_params = {}
        if init_params is not None:
            self._init_params.update(init_params)

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.datamanager.name)

        if self.port is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = get_named_client_logger(
                name=logger_name,
                port=self.port,
            )

        self.Y_optimization: Optional[Union[List, np.ndarray]] = None
        self.Y_actual_train = None

        self.budget = budget
        self.budget_type = budget_type

        # Please mypy to prevent not defined attr
        self.model = self._get_model()

    def _get_model(self) -> BaseEstimator:
        if not isinstance(self.configuration, Configuration):
            model = self.model_class(config=self.configuration,
                                     random_state=self.seed,
                                     init_params=self._init_params)
        else:
            if self.task_type in REGRESSION_TASKS:
                dataset_properties = {
                    'task': self.task_type,
                    'sparse': self.datamanager.info['is_sparse'] == 1,
                    'multioutput': self.task_type == MULTIOUTPUT_REGRESSION,
                }
            else:
                dataset_properties = {
                    'task': self.task_type,
                    'sparse': self.datamanager.info['is_sparse'] == 1,
                    'multilabel': self.task_type == MULTILABEL_CLASSIFICATION,
                    'multiclass': self.task_type == MULTICLASS_CLASSIFICATION,
                }
            model = self.model_class(config=self.configuration,
                                     dataset_properties=dataset_properties,
                                     random_state=self.seed,
                                     include=self.include,
                                     exclude=self.exclude,
                                     init_params=self._init_params)
        return model

    def _loss(self, y_true: np.ndarray, y_hat: np.ndarray,
              scoring_functions: Optional[List[Scorer]] = None
              ) -> Union[float, Dict[str, float]]:
        """Auto-sklearn follows a minimization goal.
        The calculate_loss internally translate a score function to
        a minimization problem.

        For a dummy prediction, the worst result is assumed.

        Parameters
        ----------
            y_true
        """
        scoring_functions = (
            self.scoring_functions
            if scoring_functions is None
            else scoring_functions
        )
        if not isinstance(self.configuration, Configuration):
            if scoring_functions:
                return {self.metric.name: self.metric._worst_possible_result}
            else:
                return self.metric._worst_possible_result

        return calculate_loss(
            y_true, y_hat, self.task_type, self.metric,
            scoring_functions=scoring_functions)

    def finish_up(
        self,
        loss: Union[Dict[str, float], float],
        train_loss: Optional[Union[float, Dict[str, float]]],
        opt_pred: np.ndarray,
        valid_pred: np.ndarray,
        test_pred: np.ndarray,
        additional_run_info: Optional[TYPE_ADDITIONAL_INFO],
        file_output: bool,
        final_call: bool,
        status: StatusType,
    ) -> Tuple[float, Union[float, Dict[str, float]], int,
               Dict[str, Union[str, int, float, Dict, List, Tuple]]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            file_out_loss, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred,
            )
        else:
            file_out_loss = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred,
        )

        if file_out_loss is not None:
            return self.duration, file_out_loss, self.seed, additional_run_info_

        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss_.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss

        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': status}
        if final_call:
            rval_dict['final_queue_element'] = True

        self.queue.put(rval_dict)
        return self.duration, loss_, self.seed, additional_run_info_

    def calculate_auxiliary_losses(
        self,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss: Optional[Union[float, Dict[str, float]]] = self._loss(
                    self.y_valid, Y_valid_pred)
                if isinstance(validation_loss, dict):
                    validation_loss = validation_loss[self.metric.name]
            else:
                validation_loss = None
        else:
            validation_loss = None

        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss: Optional[Union[float, Dict[str, float]]] = self._loss(
                    self.y_test, Y_test_pred)
                if isinstance(test_loss, dict):
                    test_loss = test_loss[self.metric.name]
            else:
                test_loss = None
        else:
            test_loss = None

        return validation_loss, test_loss

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Dict[str, Union[str, int, float, List, Dict, Tuple]]]:
        # Abort if self.Y_optimization is None
        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is None:
            return None, {}

        # Abort in case of shape misalignment
        if np.shape(self.Y_optimization)[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (np.shape(self.Y_optimization), Y_optimization_pred.shape)
                 },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test']
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        # Since disable_file_output can also be a list, we have to explicitly
        # compare it with True.
        if self.disable_file_output is True:
            return None, {}

        # Notice that disable_file_output==False and disable_file_output==[]
        # means the same thing here.
        if self.disable_file_output is False:
            self.disable_file_output = []

        # Here onwards, the self.disable_file_output can be treated as a list
        self.disable_file_output = cast(List, self.disable_file_output)

        # This file can be written independently of the others down bellow
        if ('y_optimization' not in self.disable_file_output):
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        models: Optional[BaseEstimator] = None
        if hasattr(self, 'models'):
            if len(self.models) > 0 and self.models[0] is not None:  # type: ignore[attr-defined]
                if ('models' not in self.disable_file_output):

                    if self.task_type in CLASSIFICATION_TASKS:
                        models = VotingClassifier(estimators=None, voting='soft', )
                    else:
                        models = VotingRegressor(estimators=None)
                    # Mypy cannot understand hasattr yet
                    models.estimators_ = self.models  # type: ignore[attr-defined]

        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model if 'model' not in self.disable_file_output else None,
            cv_model=models if 'cv_model' not in self.disable_file_output else None,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in self.disable_file_output else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in self.disable_file_output else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in self.disable_file_output else None
            ),
        )

        return None, {}

    def _predict_proba(self, X: np.ndarray, model: BaseEstimator,
                       task_type: int, Y_train: Optional[np.ndarray] = None,
                       ) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, str(category), message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict_proba(X, batch_size=1000)

        if Y_train is None:
            raise ValueError("Y_train is required for classification problems")

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X: np.ndarray, model: BaseEstimator,
                            task_type: int, Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, str(category), message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict(X)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction: np.ndarray, Y_train: np.ndarray
                                       ) -> np.ndarray:
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
