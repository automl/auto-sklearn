import numpy as np
import scipy as sp

from sklearn.metrics.classification import _check_targets


def balanced_accuracy(solution, prediction):
    y_type, solution, prediction = _check_targets(solution, prediction)

    if y_type not in ["binary", "multiclass", 'multilabel-indicator']:
        raise ValueError("{0} is not supported".format(y_type))

    if y_type == 'binary':
        # Do not transform into any multiclass representation
        pass

    elif y_type == 'multiclass':
        # Need to create a multiclass solution and a multiclass predictions
        max_class = int(np.max((np.max(solution), np.max(prediction))))
        solution_binary = np.zeros((len(solution), max_class + 1))
        prediction_binary = np.zeros((len(prediction), max_class + 1))
        for i in range(len(solution)):
            solution_binary[i, int(solution[i])] = 1
            prediction_binary[i, int(prediction[i])] = 1
        solution = solution_binary
        prediction = prediction_binary

    elif y_type == 'multilabel-indicator':
        solution = solution.toarray()
        prediction = prediction.toarray()
    else:
        raise NotImplementedError('bac_metric does not support task type %s'
                                  % y_type)

    fn = np.sum(np.multiply(solution, (1 - prediction)), axis=0,
                dtype=float)
    tp = np.sum(np.multiply(solution, prediction), axis=0, dtype=float)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)

    if y_type in ('binary', 'multilabel-indicator'):
        tn = np.sum(np.multiply((1 - solution), (1 - prediction)),
                    axis=0, dtype=float)
        fp = np.sum(np.multiply((1 - solution), prediction), axis=0,
                    dtype=float)
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
        base_bac = 0.5  # random predictions for binary case
    elif y_type == 'multiclass':
        label_num = solution.shape[1]
        bac = tpr
        base_bac = 1. / label_num  # random predictions for multiclass case
    else:
        raise ValueError(y_type)

    return np.mean(bac)  # average over all classes
    # Normalize: 0 for random, 1 for perfect
    #score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    #return score
