import typing
import warnings

import numpy as np

import scipy.sparse as sp
from scipy.sparse import coo_matrix

from sklearn.metrics import auc, roc_curve
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import column_or_1d


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Optional[typing.List] = None,
    sample_weight: typing.Optional[typing.List] = None,
    normalize: typing.Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification

    Parameters
    ----------
    y_true: np.ndarray
        An array-like list of ground truth values
    y_pred: np.ndarray,
        An array-like list of model predictions
    labels: typing.Optional[typing.List]
        List of labels to index the matrix
    sample_weight: typing.Optional[typing.List]
        Sample weights to change the score weight while reducing via average
    normalize: typing.Optional[str]
        Normalizes confusion matrix over the true (rows)

    Returns
    -------
    Confusion Matrix
    """

    if labels is None:
        n_labels = unique_labels(y_true, y_pred).size
    else:
        if np.all([label not in y_true for label in labels]):
            raise ValueError("At least one label specified must be in y_true")
        n_labels = np.asarray(labels).size

    if sample_weight is None:
        local_sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    elif np.shape(sample_weight)[0] != np.shape(y_true)[0]:
        local_sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        local_sample_weight = sample_weight

    # # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    local_sample_weight = local_sample_weight[ind]

    cm = coo_matrix((local_sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels), dtype=np.int64,
                    ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm


def BalancedAccuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: typing.Optional[typing.List] = None,
    adjusted: bool = False,
    task_type: typing.Optional[str] = None,
) -> float:
    """
    Computes the balanced accuracy
    This version honors the sklearn implementation without formatting checks to
    speed up computations.

    Parameters
    ----------
    y_true: np.ndarray
        An array-like list of ground truth values
    y_pred: np.ndarray,
        An array-like list of model predictions
    sample_weight: typing.Optional[typing.List]
        Sample weights to change the score weight while reducing via average
    adjusted: bool
        When true, the result is adjusted for chance, so that random performance
        would score 0, and perfect performance scores 1.
    task_type: str
        The task type at hand, as defined by type_of_target on the ground truth.
        It should be one of binary, multiclass or multilabel-indicator

    Returns
    -------
    The evaluation of the predictions against the ground truth
    """
    y_true = y_true.astype(np.int64)
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


def Accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    sample_weight: typing.Optional[typing.List] = None,
    task_type: typing.Optional[str] = None,
) -> float:
    """
    Computes the accuracy score
    This version honors the sklearn implementation without formatting checks to
    speed up computations.

    Parameters
    ----------
    y_true: np.ndarray
        An array-like list of ground truth values
    y_pred: np.ndarray,
        An array-like list of model predictions
    sample_weight: typing.Optional[typing.List]
        Sample weights to change the score weight while reducing via average
    normalize: bool
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    task_type: str
        The task type at hand, as defined by type_of_target on the ground truth.
        It should be one of binary, multiclass or multilabel-indicator

    Returns
    -------
    The evaluation of the predictions against the ground truth
    """
    if task_type is None:
        task_type = type_of_target(y_true)

    if task_type.startswith('multilabel'):
        differing_labels = np.count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    if normalize:
        return np.average(score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(score, sample_weight)
    else:
        return score.sum()


def label_binarize(
    y: np.ndarray,
    classes: np.ndarray,
    task_type: str,
    neg_label: int = 0,
    pos_label: int = 1
) -> np.ndarray:
    """
    Binarize labels in a one-vs-all fashion

    Parameters
    ----------
    y: np.ndarray
        An array-like list of elements to encode
    classes: np.ndarray
        A sequence of unique labels for each class
    task_type: str
        The task type at hand, as defined by type_of_target on the ground truth.
        It should be one of binary, multiclass or multilabel-indicator
    neg_label: int
        Value with which negative labels must be encoded.
    pos_label: int
        Value with which positive labels must be encoded.

    Returns
    -------
        An encoded version of y
    """
    if neg_label >= pos_label:
        raise ValueError("neg_label={0} must be strictly less than "
                         "pos_label={1}.".format(neg_label, pos_label))

    # To account for pos_label == 0 in the dense case
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label

    if 'multioutput' in task_type:
        raise ValueError("Multioutput target data is not supported with label "
                         "binarization")
    if task_type == 'unknown':
        raise ValueError("The type of target data is not known")

    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)

    if task_type.startswith('binary'):
        if n_classes == 1:
            Y = np.zeros((len(y), 1), dtype=np.int)
            Y += neg_label
            return Y
        elif len(classes) >= 3:
            task_type = "multiclass"

    sorted_class = np.sort(classes)
    if task_type.startswith('multilabel'):
        y_n_classes = y.shape[1] if hasattr(y, 'shape') else len(y[0])
        if classes.size != y_n_classes:
            raise ValueError("classes {0} mismatch with the labels {1}"
                             " found in the data"
                             .format(classes, unique_labels(y)))

    if task_type.startswith('binary') or task_type.startswith('multiclass'):
        y = column_or_1d(y)

        # pick out the known labels from y
        y_in_classes = np.in1d(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = sp.csr_matrix((data, indices, indptr),
                          shape=(n_samples, n_classes))
    elif task_type.startswith('multilabel'):
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError("%s target data is not supported with label "
                         "binarization" % task_type)

    Y = Y.toarray()
    Y = Y.astype(int, copy=False)

    if neg_label != 0:
        Y[Y == 0] = neg_label

    if pos_switch:
        Y[Y == pos_label] = 0

    # preserve label ordering
    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]

    if task_type.startswith('binary'):
        Y = Y[:, -1].reshape((-1, 1))

    return Y


def LogLoss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
    normalize: bool = True,
    sample_weight: typing.Optional[typing.List] = None,
    labels: typing.Optional[typing.List] = None,
    task_type: typing.Optional[str] = None,
) -> float:
    """
    Computes the logistic loss
    This version honors the sklearn implementation without formatting checks to
    speed up computations.

    Parameters
    ----------
    y_true: np.ndarray
        An array-like list of ground truth values
    y_pred: np.ndarray,
        An array-like list of model predictions
    eps: float
        Log loss is undefined for p=0 or p=1, so probabilities
        are clipped to max(eps, min(1 - eps, p)).
    sample_weight: typing.Optional[typing.List]
        Sample weights to change the score weight while reducing via average
    normalize: bool
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    labels: np.ndarray
        If not provided, labels will be inferred from y_true. Else labels are computed via
        label binarizer
    task_type: str
        The task type at hand, as defined by type_of_target on the ground truth.
        It should be one of binary, multiclass or multilabel-indicator

    Returns
    -------
    The evaluation of the predictions against the ground truth
    """

    if task_type is None:
        task_type = type_of_target(y_true)

    classes = unique_labels(labels if labels is not None else y_true)
    if len(classes) == 1:
        raise ValueError("The number of classes for log loss should be at least 2")

    transformed_labels = label_binarize(y=y_true, classes=classes, task_type=task_type)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    if len(classes) != y_pred.shape[1]:
        raise ValueError('The number of classes in labels is different '
                         'from that in y_pred. Classes found in '
                         'labels: {} predictions= {}'.format(classes, y_pred.shape))

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    if normalize:
        return np.average(loss, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(loss, sample_weight)
    else:
        return loss.sum()


def ROCAUC(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    sample_weight: typing.Optional[typing.List] = None,
    task_type: typing.Optional[str] = None,
    max_fpr: typing.Optional[float] = None,
    multi_class: str = "raise",
    labels: typing.Optional[typing.List] = None
) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    It does so by calculating metrics for each label, and find their unweighted
    mean. This version honors the sklearn implementation  (with the defaults average='macro')
    without formatting checks to speed up computations.

    Parameters
    ----------
    y_true: np.ndarray
        An array-like list of ground truth values
    y_pred: np.ndarray,
        An array-like list of model predictions
    average: str
        'macro' average is the only supported implementation, which is the default of
        sklearn. It is an argument to comply with roc auc api.
    sample_weight: typing.Optional[typing.List]
        Sample weights to change the score weight while reducing via average
    max_fpr: float
        None is the only supported value. Left here to comply with ROC AUC API.
    multi_class: raise
        Multiclass is not a supported implementation. As with the default sklearn implementation,
        which this method follows, passing a y_true for multiclass classification raises an error.
    task_type: str
        The task type at hand, as defined by type_of_target on the ground truth.
        It should be one of binary, multiclass or multilabel-indicator

    Returns
    -------
    The evaluation of the predictions against the ground truth
    """
    if task_type is None:
        task_type = type_of_target(y_true)

    if task_type.startswith('multiclass') or (task_type.startswith('binary') and
                                              y_score.ndim == 2 and
                                              y_score.shape[1] > 2):
        raise ValueError("Using ROC AUC for multiclass requires the user to "
                         "select a multiclass strategy as detailed in "
                         "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html ."  # noqa: E501
                         "If you still want to use this metric, kindly create "
                         "a scorer object as detailed in "
                         "https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_metrics.html#sphx-glr-examples-40-advanced-example-metrics-py"  # noqa: E501
                         )
    elif task_type.startswith('binary'):
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels, task_type=task_type)[:, 0]
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        return auc(fpr, tpr)
    elif task_type.startswith('multilabel'):
        # Calculate the metric per class, which for multilabel indicator is axis==1
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()

            fpr, tpr, _ = roc_curve(y_true_c, y_score_c, sample_weight=sample_weight)
            score[c] = auc(fpr, tpr)
        return np.average(score)
    else:
        raise ValueError("Unsupported task type {} provided".format(task_type))

    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC AUC not defined when the number of labels is less than 2")
