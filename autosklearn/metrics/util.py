# -*- encoding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy as sp
from autosklearn.constants import MULTICLASS_CLASSIFICATION, \
    BINARY_CLASSIFICATION

def sanitize_array(array):
    """
    Replace NaN and Inf (there should not be any!)
    :param array:
    :return:
    """
    a = np.ravel(array)
    maxi = np.nanmax(a[np.isfinite(a)])
    mini = np.nanmin(a[np.isfinite(a)])
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    return array



def normalize_array(solution, prediction):
    """
    Use min and max of solution as scaling factors to normalize prediction,
    then threshold it to [0, 1].

    Binarize solution to {0, 1}. This allows applying classification
    scores to all cases. In principle, this should not do anything to
    properly formatted classification inputs and outputs.

    :param solution:
    :param prediction:
    :return:
    """
    # Binarize solution
    sol = np.ravel(solution)  # convert to 1-d array
    maxi = np.nanmax(sol[np.isfinite(sol)])
    mini = np.nanmin(sol[np.isfinite(sol)])
    if maxi == mini:
        print('Warning, cannot normalize')
        return [solution, prediction]
    diff = maxi - mini
    mid = (maxi + mini) / 2.

    solution[solution >= mid] = 1
    solution[solution < mid] = 0
    # Normalize and threshold predictions (takes effect only if solution not
    # in {0, 1})

    prediction -= float(mini)
    prediction /= float(diff)

    # and if predictions exceed the bounds [0, 1]
    prediction[prediction > 1] = 1
    prediction[prediction < 0] = 0
    # Make probabilities smoother
    # new_prediction = np.power(new_prediction, (1./10))
    return [solution, prediction]


def log_loss(solution, prediction, task=BINARY_CLASSIFICATION):
    """Log loss for binary and multiclass."""
    [sample_num, label_num] = solution.shape
    # Lower gives problems with float32!
    eps = 0.00000003

    if (task == MULTICLASS_CLASSIFICATION) and (label_num > 1):
        # Make sure the lines add up to one for multi-class classification
        norma = np.sum(prediction, axis=1)
        for k in range(sample_num):
            prediction[k, :] /= sp.maximum(norma[k], eps)

        sample_num = solution.shape[0]
        for i in range(sample_num):
            j = np.argmax(solution[i, :])
            solution[i, :] = 0
            solution[i, j] = 1

        solution = solution.astype(np.int32, copy=False)
        # For the base prediction, this solution is ridiculous in the
        # multi-label case

        # Bounding of predictions to avoid log(0),1/0,...
    prediction = sp.minimum(1 - eps, sp.maximum(eps, prediction))
    # Compute the log loss
    pos_class_log_loss = -np.mean(solution * np.log(prediction), axis=0)
    if (task != MULTICLASS_CLASSIFICATION) or (label_num == 1):
        # The multi-label case is a bunch of binary problems.
        # The second class is the negative class for each column.
        neg_class_log_loss = -np.mean((1 - solution) * np.log(1 - prediction), axis=0)
        log_loss = pos_class_log_loss + neg_class_log_loss
        # Each column is an independent problem, so we average.
        # The probabilities in one line do not add up to one.
        # log_loss = mvmean(log_loss)
        # print('binary {}'.format(log_loss))
        # In the multilabel case, the right thing i to AVERAGE not sum
        # We return all the scores so we can normalize correctly later on
    else:
        # For the multiclass case the probabilities in one line add up one.
        log_loss = pos_class_log_loss
        # We sum the contributions of the columns.
        log_loss = np.sum(log_loss)
        # print('multiclass {}'.format(log_loss))
    return log_loss


def prior_log_loss(frac_pos, task=BINARY_CLASSIFICATION):
    """Baseline log loss.

    For multiplr classes ot labels return the volues for each column

    """
    eps = 1e-15
    frac_pos_ = sp.maximum(eps, frac_pos)
    if task != MULTICLASS_CLASSIFICATION:  # binary case
        frac_neg = 1 - frac_pos
        frac_neg_ = sp.maximum(eps, frac_neg)
        pos_class_log_loss_ = -frac_pos * np.log(frac_pos_)
        neg_class_log_loss_ = -frac_neg * np.log(frac_neg_)
        base_log_loss = pos_class_log_loss_ + neg_class_log_loss_
        # base_log_loss = mvmean(base_log_loss)
        # print('binary {}'.format(base_log_loss))
        # In the multilabel case, the right thing i to AVERAGE not sum
        # We return all the scores so we can normalize correctly later on
    else:  # multiclass case
        fp = frac_pos_ / sum(
            frac_pos_
        )  # Need to renormalize the lines in multiclass case
        # Only ONE label is 1 in the multiclass case active for each line
        pos_class_log_loss_ = -frac_pos * np.log(fp)
        base_log_loss = np.sum(pos_class_log_loss_)
    return base_log_loss


def binarize_predictions(array, task=BINARY_CLASSIFICATION):
    """
    Turn predictions into decisions {0,1} by selecting the class with largest
    score for multi class problems and thresh holding at 0.5 for other cases.

    :param array:
    :param task:
    :return:
    """
    # add a very small random value as tie breaker (a bit bad because
    # this changes the score every time)
    # so to make sure we get the same result every time, we seed it
    # eps = 1e-15
    # np.random.seed(sum(array.shape))
    # array = array + eps*np.random.rand(array.shape[0],array.shape[1])
    bin_array = np.zeros(array.shape, dtype=np.int32)
    if (task != MULTICLASS_CLASSIFICATION) or (array.shape[1] == 1):
        bin_array[array >= 0.5] = 1
    else:
        sample_num = array.shape[0]
        for i in range(sample_num):
            j = np.argmax(array[i, :])
            bin_array[i, j] = 1
    return bin_array


def create_multiclass_solution(solution, prediction):
    solution_binary = np.zeros((prediction.shape), dtype=np.int32)
    for i in range(solution_binary.shape[0]):
        try:
            solution_binary[i, int(solution[i])] = 1
        except IndexError as e:
            raise IndexError('too many indices to array. array has shape %s, '
                             'indices are "%s %s"' %
                             (solution_binary.shape, str(i), solution[i]))
    return solution_binary
