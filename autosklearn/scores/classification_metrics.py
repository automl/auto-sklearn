# -*- encoding: utf-8 -*-

# CLASSIFICATION METRICS (work on solutions in {0, 1} and predictions in [0, 1])
# These can be computed for regression scores only after running
# normalize_array

from __future__ import print_function

import numpy as np
import scipy as sp

from autosklearn.constants import MULTICLASS_CLASSIFICATION, \
    BINARY_CLASSIFICATION
from autosklearn.scores.common import mv_mean, binarize_predictions, \
    acc_stat, \
    tied_rank
from autosklearn.scores.util import log_loss, prior_log_loss


def acc_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Compute the accuracy.

    Get the accuracy stats
    acc = (tpr + fpr) / (tn + fp + tp + fn)
    Normalize, so 1 is the best and zero mean random...

    :param solution:
    :param prediction:
    :param task:
    :return:
    """

    label_num = solution.shape[1]
    bin_predictions = binarize_predictions(prediction, task)
    tn, fp, tp, fn = acc_stat(solution, bin_predictions)
    # Bounding to avoid division by 0
    eps = np.float(1e-15)
    tp = np.sum(tp)
    fp = np.sum(fp)
    tn = np.sum(tn)
    fn = np.sum(fn)

    if (task != MULTICLASS_CLASSIFICATION) or (label_num == 1):
        accuracy = (np.sum(tp) + np.sum(tn)) / (
            np.sum(tp) + np.sum(fp) + np.sum(tn) + np.sum(fn)
        )
    else:
        accuracy = np.sum(tp) / (np.sum(tp) + np.sum(fp))

    if (task != MULTICLASS_CLASSIFICATION) or (label_num == 1):
        base_accuracy = 0.5  # random predictions for binary case
    else:
        base_accuracy = 1. / label_num
    # Normalize: 0 for random, 1 for perfect
    score = (accuracy - base_accuracy) / sp.maximum(eps, (1 - base_accuracy))
    return score


def bac_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Compute the normalized balanced accuracy.

    The binarization and
    the normalization differ for the multi-label and multi-class case.
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    label_num = solution.shape[1]
    score = np.zeros(label_num)
    bin_prediction = binarize_predictions(prediction, task)
    [tn, fp, tp, fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)
    if (task != MULTICLASS_CLASSIFICATION) or (label_num == 1):
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
        base_bac = 0.5  # random predictions for binary case
    else:
        bac = tpr
        base_bac = 1. / label_num  # random predictions for multiclass case
    bac = mv_mean(bac)  # average over all classes
    # Normalize: 0 for random, 1 for perfect
    score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    return score


def pac_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Probabilistic Accuracy based on log_loss metric.

    We assume the solution is in {0, 1} and prediction in [0, 1].
    Otherwise, run normalize_array.
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    debug_flag = False
    [sample_num, label_num] = solution.shape
    if label_num == 1:
        task = BINARY_CLASSIFICATION
    eps = 1e-15
    the_log_loss = log_loss(solution, prediction, task)
    # Compute the base log loss (using the prior probabilities)
    pos_num = 1. * sum(solution)  # float conversion!
    frac_pos = pos_num / sample_num  # prior proba of positive class
    the_base_log_loss = prior_log_loss(frac_pos, task)
    # Alternative computation of the same thing (slower)
    # Should always return the same thing except in the multi-label case
    # For which the analytic solution makes more sense
    if debug_flag:
        base_prediction = np.empty(prediction.shape)
        for k in range(sample_num):
            base_prediction[k, :] = frac_pos
        base_log_loss = log_loss(solution, base_prediction, task)
        diff = np.array(abs(the_base_log_loss - base_log_loss))
        if len(diff.shape) > 0:
            diff = max(diff)
        if (diff) > 1e-10:
            print('Arrggh {} != {}'.format(the_base_log_loss, base_log_loss))
    # Exponentiate to turn into an accuracy-like score.
    # In the multi-label case, we need to average AFTER taking the exp
    # because it is an NL operation
    pac = mv_mean(np.exp(-the_log_loss))
    base_pac = mv_mean(np.exp(-the_base_log_loss))
    # Normalize: 0 for random, 1 for perfect
    score = (pac - base_pac) / sp.maximum(eps, (1 - base_pac))
    return score


def f1_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Compute the normalized f1 measure.

    The binarization differs
    for the multi-label and multi-class case.
    A non-weighted average over classes is taken.
    The score is normalized.
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    label_num = solution.shape[1]
    score = np.zeros(label_num)
    bin_prediction = binarize_predictions(prediction, task)
    [tn, fp, tp, fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    true_pos_num = sp.maximum(eps, tp + fn)
    found_pos_num = sp.maximum(eps, tp + fp)
    tp = sp.maximum(eps, tp)
    tpr = tp / true_pos_num  # true positive rate (recall)
    ppv = tp / found_pos_num  # positive predictive value (precision)
    arithmetic_mean = 0.5 * sp.maximum(eps, tpr + ppv)
    # Harmonic mean:
    f1 = tpr * ppv / arithmetic_mean
    # Average over all classes
    f1 = mv_mean(f1)
    # Normalize: 0 for random, 1 for perfect
    if (task != MULTICLASS_CLASSIFICATION) or (label_num == 1):
        # How to choose the "base_f1"?
        # For the binary/multilabel classification case, one may want to predict all 1.
        # In that case tpr = 1 and ppv = frac_pos. f1 = 2 * frac_pos / (1+frac_pos)
        #     frac_pos = mvmean(solution.ravel())
        #     base_f1 = 2 * frac_pos / (1+frac_pos)
        # or predict random values with probability 0.5, in which case
        #     base_f1 = 0.5
        # the first solution is better only if frac_pos > 1/3.
        # The solution in which we predict according to the class prior frac_pos gives
        # f1 = tpr = ppv = frac_pos, which is worse than 0.5 if frac_pos<0.5
        # So, because the f1 score is used if frac_pos is small (typically <0.1)
        # the best is to assume that base_f1=0.5
        base_f1 = 0.5
    # For the multiclass case, this is not possible (though it does not make much sense to
    # use f1 for multiclass problems), so the best would be to assign values at random to get
    # tpr=ppv=frac_pos, where frac_pos=1/label_num
    else:
        base_f1 = 1. / label_num
    score = (f1 - base_f1) / sp.maximum(eps, (1 - base_f1))
    return score


def auc_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Normarlized Area under ROC curve (AUC).

    Return Gini index = 2*AUC-1 for  binary classification problems.
    Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
    for the predictions. If solution and prediction are not vectors, the AUC
    of the columns of the matrices are computed and averaged (with no weight).
    The same for all classification problems (in fact it treats well only the
    binary and multilabel classification problems).
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    # auc = metrics.roc_auc_score(solution, prediction, average=None)
    # There is a bug in metrics.roc_auc_score: auc([1,0,0],[1e-10,0,0])
    # incorrect
    label_num = solution.shape[1]
    auc = np.empty(label_num)
    for k in range(label_num):
        r_ = tied_rank(prediction[:, k])
        s_ = solution[:, k]
        if sum(s_) == 0:
            print(
                'WARNING: no positive class example in class {}'.format(k + 1))
        npos = sum(s_ == 1)
        nneg = sum(s_ < 1)
        auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    return 2 * mv_mean(auc) - 1

# END CLASSIFICATION METRICS
