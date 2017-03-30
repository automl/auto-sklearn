from abc import ABCMeta, abstractmethod
from functools import partial

import sklearn.metrics
from sklearn.utils.multiclass import type_of_target

from .classification_metrics import *
from .util import *


class Scorer(object, metaclass=ABCMeta):
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def __repr__(self):
        kwargs_string = "".join([", %s=%s" % (str(k), str(v))
                                 for k, v in self._kwargs.items()])
        return ("make_scorer(%s%s%s%s)"
                % (self._score_func.__name__,
                   "" if self._sign > 0 else ", greater_is_better=False",
                   self._factory_args(), kwargs_string))

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""


class _PredictScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = np.argmax(y_pred, axis=1)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)


class _ProbaScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_pred, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

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
            y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


def make_scorer(score_func, greater_is_better=True, needs_proba=False,
                needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.

    Factory inspired by scikit-learn which wraps scoring functions to be used in
    auto-sklearn. In difference to scikit-learn, auto-sklearn always needs to
    call ``predict_proba`` in order to have predictions on a seperate validation
    set to build ensembles with.

    Paramaters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

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

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = _ProbaScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
r2_scorer = make_scorer(sklearn.metrics.r2_score)
mean_squared_error_scorer = make_scorer(sklearn.metrics.mean_squared_error,
                                        greater_is_better=False)
mean_absolute_error_scorer = make_scorer(sklearn.metrics.mean_absolute_error)
median_absolute_error_scorer = make_scorer(sklearn.metrics.median_absolute_error)

# Standard Classification Scores
accuracy_scorer = make_scorer(sklearn.metrics.accuracy_score)
f1_scorer = make_scorer(sklearn.metrics.f1_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(sklearn.metrics.roc_auc_score,
                             greater_is_better=True, needs_threshold=True)
average_precision_scorer = make_scorer(sklearn.metrics.average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(sklearn.metrics.precision_score)
recall_scorer = make_scorer(sklearn.metrics.recall_score)

# Score function for probabilistic classification
log_loss_scorer = make_scorer(sklearn.metrics.log_loss, greater_is_better=False,
                              needs_proba=True)

for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_scorer(partial(metric,
                                                        pos_label=None,
                                                        average=average))
