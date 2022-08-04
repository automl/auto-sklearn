from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import collections
from functools import partial
from itertools import product

import numpy as np
import sklearn.metrics
from sklearn.utils.multiclass import type_of_target
from smac.utils.constants import MAXINT

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    REGRESSION_TASKS,
    TASK_TYPES,
)
from autosklearn.data.target_validator import SUPPORTED_XDATA_TYPES

from .util import sanitize_array


class Scorer(object, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        score_func: Callable,
        optimum: float,
        worst_possible_result: float,
        sign: float,
        kwargs: Any,
        needs_X: bool = False,
    ) -> None:
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._needs_X = needs_X
        self._worst_possible_result = worst_possible_result
        self._sign = sign

    @abstractmethod
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        pass

    def __repr__(self) -> str:
        return self.name


class _PredictScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        X_data : array-like [n_samples x n_features]
            X data used to obtain the predictions: each row x_j corresponds to the input
             used to obtain predictions y_j

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        type_true = type_of_target(y_true)
        if (
            type_true == "binary"
            and type_of_target(y_pred) == "continuous"
            and len(y_pred.shape) == 1
        ):
            # For a pred scorer, no threshold, nor probability is required
            # If y_true is binary, and y_pred is continuous
            # it means that a rounding is necessary to obtain the binary class
            y_pred = np.around(y_pred, decimals=0)
        elif (
            len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or type_true == "continuous"
        ):
            # must be regression, all other task types would return at least
            # two probabilities
            pass
        elif type_true in ["binary", "multiclass"]:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == "multilabel-indicator":
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        elif type_true == "continuous-multioutput":
            pass
        else:
            raise ValueError(type_true)

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight
        if self._needs_X is True:
            scorer_kwargs["X_data"] = X_data

        return self._sign * self._score_func(
            y_true, y_pred, **scorer_kwargs, **self._kwargs
        )


class _ProbaScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        X_data : array-like [n_samples x n_features]
            X data used to obtain the predictions: each row x_j corresponds to the input
             used to obtain predictions y_j

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if self._score_func is sklearn.metrics.log_loss:
            n_labels_pred = np.array(y_pred).reshape((len(y_pred), -1)).shape[1]
            n_labels_test = len(np.unique(y_true))
            if n_labels_pred != n_labels_test:
                labels = list(range(n_labels_pred))
                if sample_weight is not None:
                    return self._sign * self._score_func(
                        y_true,
                        y_pred,
                        sample_weight=sample_weight,
                        labels=labels,
                        **self._kwargs,
                    )
                else:
                    return self._sign * self._score_func(
                        y_true, y_pred, labels=labels, **self._kwargs
                    )

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight
        if self._needs_X is True:
            scorer_kwargs["X_data"] = X_data

        return self._sign * self._score_func(
            y_true, y_pred, **scorer_kwargs, **self._kwargs
        )


class _ThresholdScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        X_data : array-like [n_samples x n_features]
            X data used to obtain the predictions: each row x_j corresponds to the input
             used to obtain predictions y_j

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight
        if self._needs_X is True:
            scorer_kwargs["X_data"] = X_data

        return self._sign * self._score_func(
            y_true, y_pred, **scorer_kwargs, **self._kwargs
        )


def make_scorer(
    name: str,
    score_func: Callable,
    *,
    optimum: float = 1.0,
    worst_possible_result: float = 0.0,
    greater_is_better: bool = True,
    needs_proba: bool = False,
    needs_threshold: bool = False,
    needs_X: bool = False,
    **kwargs: Any,
) -> Scorer:
    """Make a scorer from a performance metric or loss function.

    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.

    Parameters
    ----------
    name: str
        Descriptive name of the metric

    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.

    worst_possible_result : int of float, default=0
        The worst score achievable by the score function, i.e. minimum in case of
        scorer function and maximum in case of loss function.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.

    needs_X : boolean, default=False
        Whether score_func requires X in __call__ to compute a metric.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better or set
        greater_is_better to False.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )

    cls = None  # type: Any
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(
        name, score_func, optimum, worst_possible_result, sign, kwargs, needs_X=needs_X
    )


# Standard regression scores
mean_absolute_error = make_scorer(
    "mean_absolute_error",
    sklearn.metrics.mean_absolute_error,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
)
mean_squared_error = make_scorer(
    "mean_squared_error",
    sklearn.metrics.mean_squared_error,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
    squared=True,
)
root_mean_squared_error = make_scorer(
    "root_mean_squared_error",
    sklearn.metrics.mean_squared_error,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
    squared=False,
)
mean_squared_log_error = make_scorer(
    "mean_squared_log_error",
    sklearn.metrics.mean_squared_log_error,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
)
median_absolute_error = make_scorer(
    "median_absolute_error",
    sklearn.metrics.median_absolute_error,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
)

r2 = make_scorer("r2", sklearn.metrics.r2_score)

# Standard Classification Scores
accuracy = make_scorer("accuracy", sklearn.metrics.accuracy_score)
balanced_accuracy = make_scorer(
    "balanced_accuracy", sklearn.metrics.balanced_accuracy_score
)

# Score functions that need decision values
roc_auc = make_scorer(
    "roc_auc",
    sklearn.metrics.roc_auc_score,
    greater_is_better=True,
    needs_threshold=True,
)
average_precision = make_scorer(
    "average_precision", sklearn.metrics.average_precision_score, needs_threshold=True
)

# NOTE: zero_division
#
#   Specified as the explicit default, see sklearn docs:
#   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn-metrics-precision-score
precision = make_scorer(
    "precision", partial(sklearn.metrics.precision_score, zero_division=0)
)
recall = make_scorer("recall", partial(sklearn.metrics.recall_score, zero_division=0))
f1 = make_scorer("f1", partial(sklearn.metrics.f1_score, zero_division=0))

# Score function for probabilistic classification
log_loss = make_scorer(
    "log_loss",
    sklearn.metrics.log_loss,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
    needs_proba=True,
)
# TODO what about mathews correlation coefficient etc?


REGRESSION_METRICS = {
    scorer.name: scorer
    for scorer in [
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        mean_squared_log_error,
        median_absolute_error,
        r2,
    ]
}

CLASSIFICATION_METRICS = {
    scorer.name: scorer
    for scorer in [accuracy, balanced_accuracy, roc_auc, average_precision, log_loss]
}

# NOTE: zero_division
#
#   Specified as the explicit default, see sklearn docs:
#   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn-metrics-precision-score
for (base_name, sklearn_metric), average in product(
    [
        ("precision", sklearn.metrics.precision_score),
        ("recall", sklearn.metrics.recall_score),
        ("f1", sklearn.metrics.f1_score),
    ],
    ["macro", "micro", "samples", "weighted"],
):
    name = f"{base_name}_{average}"
    scorer = make_scorer(
        name, partial(sklearn_metric, pos_label=None, average=average, zero_division=0)
    )
    globals()[name] = scorer  # Adds scorer to the module scope
    CLASSIFICATION_METRICS[name] = scorer


def _validate_metrics(
    metrics: Sequence[Scorer],
    scoring_functions: Optional[List[Scorer]] = None,
) -> None:
    """
    Validate metrics given to Auto-sklearn. Raises an Exception in case of a problem.

    metrics: Sequence[Scorer]
        A list of objects that hosts a function to calculate how good the
        prediction is according to the solution.
    scoring_functions: Optional[List[Scorer]]
        A list of metrics to calculate multiple losses
    """

    to_score = list(metrics)
    if scoring_functions:
        to_score.extend(scoring_functions)

    if len(metrics) == 0:
        raise ValueError("Number of metrics to compute must be greater than zero.")

    metric_counter = collections.Counter(to_score)
    metric_names_counter = collections.Counter(metric.name for metric in to_score)
    if len(metric_counter) != len(metric_names_counter):
        raise ValueError(
            "Error in metrics passed to Auto-sklearn. A metric name was used "
            "multiple times for different metrics!"
        )


def calculate_scores(
    solution: np.ndarray,
    prediction: np.ndarray,
    task_type: int,
    metrics: Sequence[Scorer],
    *,
    X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
    scoring_functions: Optional[List[Scorer]] = None,
) -> Dict[str, float]:
    """
    Returns the scores (a magnitude that allows casting the
    optimization problem as a maximization one) for the
    given Auto-Sklearn Scorer objects.

    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metrics: Sequence[Scorer]
        A list of objects that hosts a function to calculate how good the
        prediction is according to the solution.
    X_data : array-like [n_samples x n_features]
        X data used to obtain the predictions
    scoring_functions: List[Scorer]
        A list of metrics to calculate multiple losses
    Returns
    -------
    Dict[str, float]
    """
    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    _validate_metrics(metrics=metrics, scoring_functions=scoring_functions)

    to_score = list(metrics)
    if scoring_functions:
        to_score.extend(scoring_functions)

    score_dict = dict()
    if task_type in REGRESSION_TASKS:
        for metric_ in to_score:

            try:
                score_dict[metric_.name] = _compute_single_scorer(
                    metric=metric_,
                    prediction=prediction,
                    solution=solution,
                    task_type=task_type,
                    X_data=X_data,
                )
            except ValueError as e:
                print(e, e.args[0])
                if (
                    e.args[0] == "Mean Squared Logarithmic Error cannot be used when "
                    "targets contain negative values."
                ):
                    continue
                else:
                    raise e

    else:
        for metric_ in to_score:

            # TODO maybe annotate metrics to define which cases they can
            # handle?

            try:
                score_dict[metric_.name] = _compute_single_scorer(
                    metric=metric_,
                    prediction=prediction,
                    solution=solution,
                    task_type=task_type,
                    X_data=X_data,
                )
            except ValueError as e:
                if e.args[0] == "multiclass format is not supported":
                    continue
                elif (
                    e.args[0] == "Samplewise metrics are not available "
                    "outside of multilabel classification."
                ):
                    continue
                elif (
                    e.args[0] == "Target is multiclass but "
                    "average='binary'. Please choose another average "
                    "setting, one of [None, 'micro', 'macro', 'weighted']."
                ):
                    continue
                else:
                    raise e

    return score_dict


def calculate_loss(
    solution: np.ndarray,
    prediction: np.ndarray,
    task_type: int,
    metric: Scorer,
    X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
) -> float:
    """Calculate the loss with a given metric

    Parameters
    ----------
    solution: np.ndarray
        The solutions

    prediction: np.ndarray
        The predictions generated

    task_type: int
        The task type of the problem

    metric: Scorer
        The metric to use

    X_data: Optional[SUPPORTED_XDATA_TYPES]
        X data used to obtain the predictions
    """
    losses = calculate_losses(
        solution=solution,
        prediction=prediction,
        task_type=task_type,
        metrics=[metric],
        X_data=X_data,
    )
    return losses[metric.name]


def calculate_losses(
    solution: np.ndarray,
    prediction: np.ndarray,
    task_type: int,
    metrics: Sequence[Scorer],
    *,
    X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
    scoring_functions: Optional[List[Scorer]] = None,
) -> Dict[str, float]:
    """
    Returns the losses (a magnitude that allows casting the
    optimization problem as a minimization one) for the
    given Auto-Sklearn Scorer objects.

    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metrics: Sequence[Scorer]
        A list of objects that hosts a function to calculate how good the
        prediction is according to the solution.
    X_data: Optional[SUPPORTED_XDATA_TYPES]
        X data used to obtain the predictions
    scoring_functions: List[Scorer]
        A list of metrics to calculate multiple losses

    Returns
    -------
    Dict[str, float]
        A loss function for each of the provided scorer objects
    """
    score = calculate_scores(
        solution=solution,
        prediction=prediction,
        X_data=X_data,
        task_type=task_type,
        metrics=metrics,
        scoring_functions=scoring_functions,
    )
    scoring_functions = scoring_functions if scoring_functions else []

    # we expect a dict() object for which we should calculate the loss
    loss_dict = dict()
    for metric_ in scoring_functions + list(metrics):
        # maybe metric argument is not in scoring_functions
        # TODO: When metrics are annotated with type_of_target support
        # we can remove this check
        if metric_.name not in score:
            continue
        loss_dict[metric_.name] = metric_._optimum - score[metric_.name]
    return loss_dict


def compute_single_metric(
    metric: Scorer,
    prediction: np.ndarray,
    solution: np.ndarray,
    task_type: int,
    X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
) -> float:
    """
    Returns a metric for the given Auto-Sklearn Scorer object.
    It's direction is determined by the metric itself.

    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metric: Scorer
        Object that host a function to calculate how good the
        prediction is according to the solution.
    X_data : array-like [n_samples x n_features]
        X data used to obtain the predictions

    Returns
    -------
    float
    """
    score = _compute_single_scorer(
        solution=solution,
        prediction=prediction,
        metric=metric,
        X_data=X_data,
        task_type=task_type,
    )
    return metric._sign * score


def _compute_single_scorer(
    metric: Scorer,
    prediction: np.ndarray,
    solution: np.ndarray,
    task_type: int,
    X_data: Optional[SUPPORTED_XDATA_TYPES] = None,
) -> float:
    """
    Returns a score (a magnitude that allows casting the
    optimization problem as a maximization one) for the
    given Auto-Sklearn Scorer object

    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metric: Scorer
        Object that host a function to calculate how good the
        prediction is according to the solution.
    X_data : array-like [n_samples x n_features]
        X data used to obtain the predictions
    Returns
    -------
    float
    """
    if metric._needs_X:
        if X_data is None:
            raise ValueError(
                f"Metric {metric.name} needs X_data, but X_data is {X_data}"
            )
        elif X_data.shape[0] != solution.shape[0]:
            raise ValueError(
                f"X_data has wrong length. "
                f"Should be {solution.shape[0]}, but is {X_data.shape[0]}"
            )
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            score = metric(solution, cprediction, X_data=X_data)
        else:
            score = metric(solution, prediction, X_data=X_data)
        return score

    if task_type in REGRESSION_TASKS:
        # TODO put this into the regression metric itself
        cprediction = sanitize_array(prediction)
        score = metric(solution, cprediction)
    else:
        score = metric(solution, prediction)
    return score


# Must be at bottom so all metrics are defined
default_metric_for_task: Dict[int, Scorer] = {
    BINARY_CLASSIFICATION: CLASSIFICATION_METRICS["accuracy"],
    MULTICLASS_CLASSIFICATION: CLASSIFICATION_METRICS["accuracy"],
    MULTILABEL_CLASSIFICATION: CLASSIFICATION_METRICS["f1_macro"],
    REGRESSION: REGRESSION_METRICS["r2"],
    MULTIOUTPUT_REGRESSION: REGRESSION_METRICS["r2"],
}
