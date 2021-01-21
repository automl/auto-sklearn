from abc import ABCMeta
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import sklearn.metrics
from sklearn.utils.multiclass import type_of_target

from smac.utils.constants import MAXINT

from autosklearn.constants import REGRESSION_TASKS, TASK_TYPES, TASK_TYPES_TO_STRING
from autosklearn.metrics.custom_metrics import (
    Accuracy,
    BalancedAccuracy,
    LogLoss,
    ROCAUC,
)


from .util import sanitize_array


class Scorer(object, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        score_func: Callable,
        optimum: float,
        worst_possible_result: float,
        sign: float,
        needs_proba: bool,
        needs_threshold: bool,
        format_predictions: bool,
        kwargs: Any
    ) -> None:
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._worst_possible_result = worst_possible_result
        self._sign = sign
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.format_predictions = format_predictions

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Calls the scoring function to evaluate the performance of y_pred against
        y_true.

        Parameters
        ----------
        y_true: np.ndarray
            The golden truth used to evaluate the predictions
        y_pred: np.ndarray
            The predictions of the model being evaluated
        sample_weight: Optional[List[float]]
            Sample weights
        Returns
            score: float
                The evaluation of how good y_pred is as compared to y_true
        """

        if self._score_func in (sklearn.metrics.log_loss, LogLoss):
            self._kwargs.pop('labels', None)
            n_labels_pred = np.array(y_pred).reshape((len(y_pred), -1)).shape[1]
            n_labels_test = len(np.unique(y_true))
            if n_labels_pred != n_labels_test and not type_of_target(y_true
                                                                     ).startswith('multilabel'):
                # If the task type is multilabel, the label binarizer inside log loss
                # has to be fitted with multilabel data, else we run into error:
                # The object was not fitted with multilabel input.
                self._kwargs['labels'] = list(range(n_labels_pred))

        if self.format_predictions:
            y_pred = clean_predictions(self, y_pred, type_of_target(y_true))

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    def __repr__(self) -> str:
        return self.name


def clean_predictions(
    metric: Scorer,
    y_pred: np.ndarray,
    task_type: str,
) -> np.ndarray:
    """Performs prediction cleanup before calling the actual scorer function

    Parameters
    ----------
    metric: Scorer
        A wrapper over a metric calculation function
    y_pred : array-like, [n_samples x n_classes]
        Model predictions
    task_type: str
        One of
        binary
        multiclass
        multilable-indicator
        continuous
        continuous-multioutput

    Returns
    -------
    y_pred : array-like, [n_samples x n_classes]
        A post processed version of the y_pred
    """

    if metric.needs_threshold:
        if 'binary' not in task_type and 'multilabel' not in task_type:
            raise ValueError("{0} format is not supported".format(task_type))

        if task_type.startswith("binary"):
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T
    elif metric.needs_proba:
        pass
    else:
        if task_type.startswith('binary') and type_of_target(y_pred) == 'continuous' and \
                len(y_pred.shape) == 1:
            # For a pred scorer, no threshold, nor probability is required
            # If y_true is binary, and y_pred is continuous
            # it means that a rounding is necessary to obtain the binary class
            y_pred = np.around(y_pred, decimals=0)
        elif len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or \
                task_type.startswith('continuous'):
            # must be regression, all other task types would return at least
            # two probabilities
            pass
        elif task_type.startswith('binary') or task_type.startswith('multiclass'):
            y_pred = np.argmax(y_pred, axis=1)
        elif task_type.startswith('multilabel'):
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        elif task_type.startswith('continuous-multioutput'):
            pass
        else:
            raise ValueError(task_type)
    return y_pred


def make_scorer(
    name: str,
    score_func: Callable,
    optimum: float = 1.0,
    worst_possible_result: float = 0.0,
    greater_is_better: bool = True,
    needs_proba: bool = False,
    needs_threshold: bool = False,
    needs_format_predictions: bool = True,
    **kwargs: Any
) -> Scorer:
    """Make a scorer from a performance metric or loss function.

    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.

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

    needs_format_predictions: boolean, default = True,
        By default, Auto-sklearn predictions require pre-processing before a score can
        be calculated. For example, when doing binary predictions, and the later is a 2D
        array, it likely means that argmax will provide the actual prediction. A scorer
        can set this flag to false to accelerate calculation under the assumption that
        the predictions have been format already

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    return Scorer(
        name, score_func, optimum, worst_possible_result, sign,
        needs_proba, needs_threshold, needs_format_predictions, kwargs)


# Standard regression scores
mean_absolute_error = make_scorer('mean_absolute_error',
                                  sklearn.metrics.mean_absolute_error,
                                  optimum=0,
                                  worst_possible_result=MAXINT,
                                  greater_is_better=False)
mean_squared_error = make_scorer('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 optimum=0,
                                 worst_possible_result=MAXINT,
                                 greater_is_better=False,
                                 squared=True)
root_mean_squared_error = make_scorer('root_mean_squared_error',
                                      sklearn.metrics.mean_squared_error,
                                      optimum=0,
                                      worst_possible_result=MAXINT,
                                      greater_is_better=False,
                                      squared=False)
mean_squared_log_error = make_scorer('mean_squared_log_error',
                                     sklearn.metrics.mean_squared_log_error,
                                     optimum=0,
                                     worst_possible_result=MAXINT,
                                     greater_is_better=False,)
median_absolute_error = make_scorer('median_absolute_error',
                                    sklearn.metrics.median_absolute_error,
                                    optimum=0,
                                    worst_possible_result=MAXINT,
                                    greater_is_better=False)
r2 = make_scorer('r2',
                 sklearn.metrics.r2_score)

# Standard Classification Scores
accuracy = make_scorer('accuracy',
                       sklearn.metrics.accuracy_score,
                       )
balanced_accuracy = make_scorer('balanced_accuracy',
                                sklearn.metrics.balanced_accuracy_score,
                                )
f1 = make_scorer('f1',
                 sklearn.metrics.f1_score)

# Score functions that need decision values
roc_auc = make_scorer('roc_auc',
                      sklearn.metrics.roc_auc_score,
                      greater_is_better=True,
                      needs_threshold=True,
                      )
average_precision = make_scorer('average_precision',
                                sklearn.metrics.average_precision_score,
                                needs_threshold=True)
precision = make_scorer('precision',
                        sklearn.metrics.precision_score)
recall = make_scorer('recall',
                     sklearn.metrics.recall_score)

# Score function for probabilistic classification
log_loss = make_scorer('log_loss',
                       sklearn.metrics.log_loss,
                       optimum=0,
                       worst_possible_result=MAXINT,
                       greater_is_better=False,
                       needs_proba=True)
# TODO what about mathews correlation coefficient etc?

CUSTOM_METRICS = {
    'accuracy': make_scorer('accuracy', Accuracy, needs_format_predictions=False),
    'balanced_accuracy': make_scorer('balanced_accuracy', BalancedAccuracy,
                                     needs_format_predictions=False),
    'roc_auc': make_scorer('roc_auc', ROCAUC, needs_threshold=True,
                           needs_format_predictions=False),
    'log_loss': make_scorer('log_loss', LogLoss, optimum=0, worst_possible_result=MAXINT,
                            greater_is_better=False, needs_proba=True,
                            needs_format_predictions=False)
}

REGRESSION_METRICS = dict()
for scorer in [mean_absolute_error, mean_squared_error, root_mean_squared_error,
               mean_squared_log_error, median_absolute_error, r2]:
    REGRESSION_METRICS[scorer.name] = scorer

CLASSIFICATION_METRICS = dict()

for scorer in [accuracy, balanced_accuracy, roc_auc, average_precision,
               log_loss]:
    CLASSIFICATION_METRICS[scorer.name] = scorer

for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_scorer(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_scorer(qualified_name,
                                                partial(metric,
                                                        pos_label=None,
                                                        average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]


def calculate_score(
    solution: np.ndarray,
    prediction: np.ndarray,
    task_type: int,
    metric: Scorer,
    scoring_functions: Optional[List[Scorer]] = None,
) -> Union[float, Dict[str, float]]:
    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if scoring_functions:
        score_dict = dict()
        # TODO put this into the regression metric itself
        # TODO maybe annotate metrics to define which cases they can
        # handle?
        for metric_ in scoring_functions:

            try:
                score_dict[metric_.name] = get_metric_score(metric_, prediction, solution,
                                                            task_type)
            except ValueError as e:
                if task_type in REGRESSION_TASKS:
                    print(e, e.args[0])
                    if e.args[0] == "Mean Squared Logarithmic Error cannot be used when " \
                            "targets contain negative values.":
                        continue
                else:
                    if e.args[0] == 'multiclass format is not supported':
                        continue
                    elif e.args[0] == "Samplewise metrics are not available "\
                            "outside of multilabel classification.":
                        continue
                    elif e.args[0] == "Target is multiclass but "\
                            "average='binary'. Please choose another average "\
                            "setting, one of [None, 'micro', 'macro', 'weighted'].":
                        continue
                    else:
                        raise e

        if metric.name not in score_dict.keys():
            score_dict[metric.name] = get_metric_score(metric, prediction, solution, task_type)
        return score_dict

    else:
        return get_metric_score(metric, prediction, solution, task_type)


def get_metric_score(
        metric_: Scorer,
        prediction: np.ndarray,
        solution: np.ndarray,
        task_type: int,
) -> float:
    if metric_._score_func in CUSTOM_METRICS:
        # Calculating type of target has a noticeable duration
        # due to internal castings. Custom metrics support taking the pre-computed
        # type
        metric_._kwargs['task_type'] = TASK_TYPES_TO_STRING[task_type]

    if task_type in REGRESSION_TASKS:
        # TODO put this into the regression metric itself
        cprediction = sanitize_array(prediction)
        score = metric_(solution, cprediction)
    else:
        score = metric_(solution, prediction)
    return score
