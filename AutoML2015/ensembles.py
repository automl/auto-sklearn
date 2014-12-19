'''
Created on Dec 19, 2014

@author: Aaron Klein
'''

import os
import sys
import cma
import numpy as np

from AutoML2015.data import data_io


def weighted_ensemble_error(weights, *args):
    predicitons = args[0]
    true_labels = args[1]
    n_points = predicitons.shape[1]
    n_classes = predicitons.shape[2]

    weighted_predictions = np.zeros([n_points, n_classes])

    # We optimize w' = w/sum(w) instead of w to make sure that the weights are in [0,1]
    weights_norm = np.array(weights).sum()

    for i, p in enumerate(predicitons):

        weight_prime = weights[i] / weights_norm
        weighted_predictions += weight_prime * p

    # Check how many predictions are correct
    acc = float(np.count_nonzero(true_labels.T[0] == np.argmax(weighted_predictions, axis=1))) / n_points

    return 1 - acc


def weighted_ensemble(predictions, true_labels):

    n_models = predictions.shape[0]
    weights = np.ones([n_models]) / n_models
    if n_models > 1:
        res = cma.fmin(weighted_ensemble_error, weights, sigma0=0.25, args=(predictions, true_labels), options={'bounds': [0, 1]})
        weights = np.array(res[0])
    else:
        # Python-CMA does not work in a 1-D space
        weights = np.ones([1])
    weights = weights / weights.sum()
    print "weights: " + str(weights)

    return weights


def ensemble_prediction(all_predictions, weights):
    for i, w in enumerate(weights):
        all_predictions[i] *= w

    return all_predictions.mean(axis=0)


def main(data_dir, basename):
    index_run = 0
    while True:
        #=== Load the true labels of the validation data
        true_labels = np.load(os.path.join(data_dir, "true_labels.npy"))

        #=== Load the predictions from the models
        all_predictions_train = []
        dir = os.path.join(data_dir, "predictions_ensemble/")
        for f in os.listdir(dir):
            print f
            predictions = np.load(os.path.join(dir, f))
            all_predictions_train.append(predictions)

        #=== Compute the weights for the ensemble
        weights = weighted_ensemble(np.array(all_predictions_train), true_labels)

        all_predictions_valid = []
        dir = os.path.join(data_dir, "predictions_valid/")
        for f in os.listdir(dir):
            predictions = np.load(os.path.join(dir, f))
            all_predictions_valid.append(predictions)

        #=== Compute the ensemble predictions for the valid data
        Y_test = ensemble_prediction(np.array(all_predictions_valid), weights)

        filename_test = basename + '_valid_' + str(index_run) + '.predict'
        data_io.write(os.path.join(data_dir, filename_test), Y_test)

        all_predictions_test = []
        dir = os.path.join(data_dir, "predictions_test/")
        for f in os.listdir(dir):
            predictions = np.load(os.path.join(dir, f))
            all_predictions_test.append(predictions)

        #=== Compute the ensemble predictions for the test data
        Y_test = ensemble_prediction(np.array(all_predictions_test), weights)

        filename_test = basename + '_test_' + str(index_run) + '.predict'
        data_io.write(os.path.join(data_dir, filename_test), Y_test)
        index_run += 1
        break
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
