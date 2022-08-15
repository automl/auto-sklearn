from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Type, Union, cast

import logging
import multiprocessing
import time
import warnings

import numpy as np
from ConfigSpace import Configuration
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from smac.tae import StatusType
from threadpoolctl import threadpool_limits

import autosklearn.pipeline.classification
import autosklearn.pipeline.regression
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION_TASKS,
)
from autosklearn.data.target_validator import (
    SUPPORTED_TARGET_TYPES,
    SUPPORTED_XDATA_TYPES,
)
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.components.base import ThirdPartyComponents, _addons
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from autosklearn.util.logging_ import PicklableClientLogger, get_named_client_logger

# General TYPE definitions for numpy
TYPE_ADDITIONAL_INFO = Dict[str, Union[int, float, str, Dict, List, Tuple]]


class MyDummyClassifier(DummyClassifier):
    def __init__(
        self,
        config: Configuration,
        random_state: Optional[Union[int, np.random.RandomState]],
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super().__init__(strategy="uniform")
        else:
            super().__init__(strategy="most_frequent")

        self.random_state = random_state
        self.init_params = init_params
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude
        self.feat_type = feat_type

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[Union[np.ndarray, List]] = None,
    ) -> DummyClassifier:
        return super(MyDummyClassifier, self).fit(
            np.ones((X.shape[0], 1)), y, sample_weight=sample_weight
        )

    def fit_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> DummyClassifier:
        return self.fit(X, y)

    def predict_proba(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        probas = super().predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(np.float32)
        return probas

    def estimator_supports_iterative_fit(self) -> bool:
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:
        return None


class MyDummyRegressor(DummyRegressor):
    def __init__(
        self,
        config: Configuration,
        random_state: Optional[Union[int, np.random.RandomState]],
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super().__init__(strategy="mean")
        else:
            super().__init__(strategy="median")
        self.random_state = random_state
        self.init_params = init_params
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude
        self.feat_type = feat_type

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[Union[np.ndarray, List]] = None,
    ) -> DummyRegressor:
        return super().fit(np.ones((X.shape[0], 1)), y, sample_weight=sample_weight)

    def fit_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> DummyRegressor:
        return self.fit(X, y)

    def predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super().predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:
        return None


def _fit_and_suppress_warnings(
    logger: Union[logging.Logger, PicklableClientLogger],
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
) -> BaseEstimator:
    def send_warnings_to_log(
        message: Union[Warning, str],
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: Optional[TextIO] = None,
        line: Optional[str] = None,
    ) -> None:
        logger.debug("%s:%s: %s:%s" % (filename, lineno, str(category), message))
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
        metrics: Sequence[Scorer],
        additional_components: Dict[str, ThirdPartyComponents],
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

        # Limit the number of threads that numpy uses
        threadpool_limits(limits=1)

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.port = port
        self.queue = queue

        self.datamanager = self.backend.load_datamanager()
        self.feat_type = self.datamanager.feat_type
        self.include = include
        self.exclude = exclude

        self.X_test = self.datamanager.data.get("X_test")
        self.y_test = self.datamanager.data.get("Y_test")

        self.metrics = metrics
        self.task_type = self.datamanager.info["task"]
        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization
        self.scoring_functions = scoring_functions if scoring_functions else []

        if isinstance(disable_file_output, (bool, list)):
            self.disable_file_output: Union[bool, List[str]] = disable_file_output
        else:
            raise ValueError("disable_file_output should be either a bool or a list")

        if self.task_type in REGRESSION_TASKS:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyRegressor
            else:
                self.model_class = (
                    autosklearn.pipeline.regression.SimpleRegressionPipeline
                )
            self.predict_function = self._predict_regression
        else:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyClassifier
            else:
                self.model_class = (
                    autosklearn.pipeline.classification.SimpleClassificationPipeline
                )
            self.predict_function = self._predict_proba

        self._init_params = {"data_preprocessor:feat_type": self.datamanager.feat_type}

        if init_params is not None:
            self._init_params.update(init_params)

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        logger_name = "%s(%d):%s" % (
            self.__class__.__name__.split(".")[-1],
            self.seed,
            self.datamanager.name,
        )

        if self.port is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = get_named_client_logger(
                name=logger_name,
                port=self.port,
            )

        self.X_optimization: Optional[SUPPORTED_XDATA_TYPES] = None
        self.Y_optimization: Optional[SUPPORTED_TARGET_TYPES] = None
        self.Y_actual_train = None

        self.budget = budget
        self.budget_type = budget_type

        # Add 3rd-party components to the list of 3rd-party components in case this
        # wasn't done before (this happens if we run in parallel and the components
        # are only passed to the AbstractEvaluator via the TAE and are not there
        # yet because the worker is in its own process).
        for key in additional_components:
            for component_name, component in additional_components[
                key
            ].components.items():
                if component_name not in _addons[key].components:
                    _addons[key].add_component(component)

        # Please mypy to prevent not defined attr
        self.model = self._get_model(feat_type=self.feat_type)

    def _get_model(self, feat_type: Optional[FEAT_TYPE_TYPE]) -> BaseEstimator:
        if not isinstance(self.configuration, Configuration):
            model = self.model_class(
                feat_type=feat_type,
                config=self.configuration,
                random_state=self.seed,
                init_params=self._init_params,
            )
        else:
            if self.task_type in REGRESSION_TASKS:
                dataset_properties = {
                    "task": self.task_type,
                    "sparse": self.datamanager.info["is_sparse"] == 1,
                    "multioutput": self.task_type == MULTIOUTPUT_REGRESSION,
                }
            else:
                dataset_properties = {
                    "task": self.task_type,
                    "sparse": self.datamanager.info["is_sparse"] == 1,
                    "multilabel": self.task_type == MULTILABEL_CLASSIFICATION,
                    "multiclass": self.task_type == MULTICLASS_CLASSIFICATION,
                }
            model = self.model_class(
                feat_type=feat_type,
                config=self.configuration,
                dataset_properties=dataset_properties,
                random_state=self.seed,
                include=self.include,
                exclude=self.exclude,
                init_params=self._init_params,
            )
        return model

    def _loss(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
    ) -> Dict[str, float]:
        """Auto-sklearn follows a minimization goal.
        The calculate_loss internally translate a score function to
        a minimization problem.

        For a dummy prediction, the worst result is assumed.

        Parameters
        ----------
            y_true
        """
        return calculate_losses(
            y_true,
            y_hat,
            self.task_type,
            self.metrics,
            X_data=X_data,
            scoring_functions=self.scoring_functions,
        )

    def finish_up(
        self,
        loss: Union[Dict[str, float], float],
        train_loss: Optional[Dict[str, float]],
        opt_pred: np.ndarray,
        test_pred: np.ndarray,
        additional_run_info: Optional[TYPE_ADDITIONAL_INFO],
        file_output: bool,
        final_call: bool,
        status: StatusType,
    ) -> Tuple[
        float,
        Union[float, Dict[str, float]],
        int,
        Dict[str, Union[str, int, float, Dict, List, Tuple]],
    ]:
        """Do everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)
        """
        self.duration = time.time() - self.starttime

        if file_output:
            file_out_loss, additional_run_info_ = self.file_output(opt_pred, test_pred)
        else:
            file_out_loss = None
            additional_run_info_ = {}

        test_loss = self.calculate_auxiliary_losses(test_pred)

        if file_out_loss is not None:
            return self.duration, file_out_loss, self.seed, additional_run_info_

        loss_ = loss
        for metric in self.metrics:
            if metric.name not in loss_:
                raise ValueError(
                    f"Unable to compute optimization metric {metric.name}. Are you "
                    f"sure {metric.name} is applicable for the given task type?"
                )
        if len(self.metrics) == 1:
            loss = loss_[self.metrics[0].name]
        else:
            loss = {metric.name: loss_[metric.name] for metric in self.metrics}

        additional_run_info = {} if additional_run_info is None else additional_run_info
        for metric in self.scoring_functions:
            if metric.name in loss_:
                additional_run_info[metric.name] = loss_[metric.name]
        additional_run_info["duration"] = self.duration
        additional_run_info["num_run"] = self.num_run
        if train_loss is not None:
            if len(self.metrics) == 1:
                additional_run_info["train_loss"] = train_loss[self.metrics[0].name]
            else:
                additional_run_info["train_loss"] = [
                    train_loss[metric.name] for metric in self.metrics
                ]
        if test_loss is not None:
            additional_run_info["test_loss"] = test_loss

        return_value_dict = {
            "loss": loss,
            "additional_run_info": additional_run_info,
            "status": status,
        }
        if final_call:
            return_value_dict["final_queue_element"] = True

        self.queue.put(return_value_dict)
        return self.duration, loss_, self.seed, additional_run_info_

    def calculate_auxiliary_losses(
        self,
        Y_test_pred: np.ndarray | None,
    ) -> float | dict[str, float] | None:
        if Y_test_pred is None or self.y_test is None:
            return None

        test_loss = self._loss(self.y_test, Y_test_pred)
        if len(self.metrics) == 1:
            test_loss = test_loss[self.metrics[0].name]

        return test_loss

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> tuple[float | None, dict[str, Any]]:
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
                    "error": "Targets %s and prediction %s don't have "
                    "the same length. Probably training didn't "
                    "finish"
                    % (np.shape(self.Y_optimization), Y_optimization_pred.shape)
                },
            )

        # Abort if predictions contain NaNs
        for y, s in [(Y_optimization_pred, "optimization"), (Y_test_pred, "test")]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {"error": "Model predictions for %s set contains NaNs." % s},
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
        if "y_optimization" not in self.disable_file_output:
            if self.output_y_hat_optimization:
                self.backend.save_additional_data(
                    self.Y_optimization, what="targets_ensemble"
                )
                self.backend.save_additional_data(
                    self.X_optimization, what="input_ensemble"
                )

        models: Optional[BaseEstimator] = None
        if hasattr(self, "models"):
            if len(self.models) > 0 and self.models[0] is not None:
                if "models" not in self.disable_file_output:

                    if self.task_type in CLASSIFICATION_TASKS:
                        models = VotingClassifier(
                            estimators=None,
                            voting="soft",
                        )
                    else:
                        models = VotingRegressor(estimators=None)
                    # Mypy cannot understand hasattr yet
                    models.estimators_ = self.models  # type: ignore[attr-defined]

        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model if "model" not in self.disable_file_output else None,
            cv_model=models if "cv_model" not in self.disable_file_output else None,
            # TODO: below line needs to be deleted once backend is updated
            valid_predictions=None,
            ensemble_predictions=(
                Y_optimization_pred
                if "y_optimization" not in self.disable_file_output
                else None
            ),
            test_predictions=(
                Y_test_pred if "y_test" not in self.disable_file_output else None
            ),
        )

        return None, {}

    def _predict_proba(
        self,
        X: np.ndarray,
        model: BaseEstimator,
        task_type: int,
        Y_train: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug(
                "%s:%s: %s:%s" % (filename, lineno, str(category), message)
            )
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict_proba(X, batch_size=1000)

        if Y_train is None:
            raise ValueError("Y_train is required for classification problems")

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(
        self,
        X: np.ndarray,
        model: BaseEstimator,
        task_type: int,
        Y_train: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug(
                "%s:%s: %s:%s" % (filename, lineno, str(category), message)
            )
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict(X)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(
        self, prediction: np.ndarray, Y_train: np.ndarray
    ) -> np.ndarray:
        num_classes = self.datamanager.info["label_num"]

        if (
            self.task_type == MULTICLASS_CLASSIFICATION
            and prediction.shape[1] < num_classes
        ):
            if Y_train is None:
                raise ValueError("Y_train must not be None!")
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros(
                (prediction.shape[0], num_classes), dtype=np.float32
            )

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
