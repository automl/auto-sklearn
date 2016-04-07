# -*- encoding: utf-8 -*-

# CLASSIFICATION METRICS (work on solutions in {0, 1} and predictions in [0, 1])
# These can be computed for regression scores only after running
# normalize_array

from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.stats
from autosklearn.constants import MULTICLASS_CLASSIFICATION, \
    BINARY_CLASSIFICATION, METRIC_TO_STRING, MULTILABEL_CLASSIFICATION
from autosklearn.metrics.util import log_loss, prior_log_loss, \
    binarize_predictions, normalize_array, create_multiclass_solution


def calculate_score(metric, solution, prediction, task):
    if solution.shape[0] != prediction.shape[0]:
        raise ValueError('Solution and prediction have different number of '
                         'samples: %d and %d' % (solution.shape[0],
                                                 prediction.shape[0]))

    metric = METRIC_TO_STRING[metric]
    return globals()[metric](solution, prediction, task)



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
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)

        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            # Prediction will be copied into a new binary array - no copy
            prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape ) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)

    elif task == MULTILABEL_CLASSIFICATION:
        pass
    else:
        raise NotImplementedError('acc_metric does not support task type %s'
                                  % task)

    bin_predictions = binarize_predictions(prediction, task)

    tn = np.sum(np.multiply((1 - solution), (1 - bin_predictions)), axis=0,
                dtype=float)
    fn = np.sum(np.multiply(solution, (1 - bin_predictions)), axis=0,
                dtype=float)
    tp = np.sum(np.multiply(solution, bin_predictions), axis=0,
                dtype=float)
    fp = np.sum(np.multiply((1 - solution), bin_predictions), axis=0,
                dtype=float)
    # Bounding to avoid division by 0, 1e-7 because of float32
    eps = np.float(1e-7)
    """
    tp = np.sum(tp)
    fp = np.sum(fp)
    tn = np.sum(tn)
    fn = np.sum(fn)
   """
    if task in (BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION):
       accuracy = (np.sum(tp) + np.sum(tn)) / (
            np.sum(tp) + np.sum(fp) + np.sum(tn) + np.sum(fn)
        )
    elif task == MULTICLASS_CLASSIFICATION:
        accuracy = np.sum(tp) / (np.sum(tp) + np.sum(fp))

    if task in (BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION):
        base_accuracy = 0.5  # random predictions for binary case
    elif task == MULTICLASS_CLASSIFICATION:
        label_num = solution.shape[1]
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
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)

        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            # Prediction will be copied into a new binary array - no copy
            prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
    elif task == MULTILABEL_CLASSIFICATION:
        pass
    else:
        raise NotImplementedError('bac_metric does not support task type %s'
                                  % task)
    bin_prediction = binarize_predictions(prediction, task)


    fn = np.sum(np.multiply(solution, (1 - bin_prediction)), axis=0,
                dtype=float)
    tp = np.sum(np.multiply(solution, bin_prediction), axis=0, dtype=float)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)

    if task in (BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION):
        tn = np.sum(np.multiply((1 - solution), (1 - bin_prediction)),
                    axis=0, dtype=float)
        fp = np.sum(np.multiply((1 - solution), bin_prediction), axis=0,
                    dtype=float)
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
        base_bac = 0.5  # random predictions for binary case
    elif task == MULTICLASS_CLASSIFICATION:
        label_num = solution.shape[1]
        bac = tpr
        base_bac = 1. / label_num  # random predictions for multiclass case

    bac = np.mean(bac)  # average over all classes
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
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
        solution = solution.copy()

        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            # Prediction will be copied into a new binary array - no copy
            prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
    elif task == MULTILABEL_CLASSIFICATION:
        solution = solution.copy()
    else:
        raise NotImplementedError('auc_metric does not support task type %s'
                                  % task)
    solution, prediction = normalize_array(solution, prediction.copy())

    [sample_num, label_num] = solution.shape
    if label_num == 1:
        task = BINARY_CLASSIFICATION
    eps = 1e-7
    # Compute the base log loss (using the prior probabilities)
    pos_num = 1. * np.sum(solution, axis=0, dtype=float)  # float conversion!
    frac_pos = pos_num / sample_num  # prior proba of positive class
    the_base_log_loss = prior_log_loss(frac_pos, task)
    the_log_loss = log_loss(solution, prediction, task)

    # Exponentiate to turn into an accuracy-like score.
    # In the multi-label case, we need to average AFTER taking the exp
    # because it is an NL operation
    pac = np.mean(np.exp(-the_log_loss))
    base_pac = np.mean(np.exp(-the_base_log_loss))
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
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)

        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            # Prediction will be copied into a new binary array - no copy
            prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
    elif task == MULTILABEL_CLASSIFICATION:
        pass
    else:
        raise NotImplementedError('f1_metric does not support task type %s'
                                  % task)
    bin_prediction = binarize_predictions(prediction, task)

    # Bounding to avoid division by 0
    eps = 1e-15
    fn = np.sum(np.multiply(solution, (1 - bin_prediction)), axis=0, dtype=float)
    tp = np.sum(np.multiply(solution, bin_prediction), axis=0, dtype=float)
    fp = np.sum(np.multiply((1 - solution), bin_prediction), axis=0, dtype=float)
    true_pos_num = sp.maximum(eps, tp + fn)
    found_pos_num = sp.maximum(eps, tp + fp)
    tp = sp.maximum(eps, tp)
    tpr = tp / true_pos_num  # true positive rate (recall)
    ppv = tp / found_pos_num  # positive predictive value (precision)
    arithmetic_mean = 0.5 * sp.maximum(eps, tpr + ppv)
    # Harmonic mean:
    f1 = tpr * ppv / arithmetic_mean
    # Average over all classes
    f1 = np.mean(f1)
    # Normalize: 0 for random, 1 for perfect
    if task in (BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION):
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
    elif task == MULTICLASS_CLASSIFICATION:
        label_num = solution.shape[1]
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
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
        solution = solution.copy()

        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            elif prediction.shape[1] == 2:
                # Prediction will be copied into a new binary array - no copy
                prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
    elif task == MULTILABEL_CLASSIFICATION:
        solution = solution.copy()
    else:
        raise NotImplementedError('auc_metric does not support task type %s'
                                  % task)

    solution, prediction = normalize_array(solution, prediction.copy())

    label_num = solution.shape[1]
    auc = np.empty(label_num)
    for k in range(label_num):
        r_ = scipy.stats.rankdata(prediction[:, k])
        s_ = solution[:, k]
        if sum(s_) == 0:
            print(
                'WARNING: no positive class example in class {}'.format(k + 1))
        npos = np.sum(s_ == 1)
        nneg = np.sum(s_ < 1)
        auc[k] = (np.sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    auc[~np.isfinite(auc)] = 0
    return 2 * np.mean(auc) - 1

