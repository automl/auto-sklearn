'''
Created on Dec 19, 2014

@author: Aaron Klein
'''

import os
import sys
import cma
import numpy as np
import time

from data import data_io
from models import evaluate
import util.Stopwatch


def weighted_ensemble_error(weights, *args):
    predictions = args[0]
    true_labels = args[1]
    metric = args[2]
    task_type = args[3]

    weight_prime = weights / weights.sum()
    weighted_predictions = ensemble_prediction(predictions, weight_prime)

    score = evaluate.calculate_score(true_labels, weighted_predictions,
                                     task_type, metric)
    return 1 - score


def weighted_ensemble(predictions, true_labels, info):

    n_models = predictions.shape[0]
    weights = np.ones([n_models]) / n_models
    if n_models > 1:
        res = cma.fmin(weighted_ensemble_error, weights, sigma0=0.25,
                       args=(predictions, true_labels, info['metric'],
                             info['task']), options={'bounds': [0, 1]})
        weights = np.array(res[0])
    else:
        # Python-CMA does not work in a 1-D space
        weights = np.ones([1])
    weights = weights / weights.sum()

    return weights


def ensemble_prediction(all_predictions, weights):
    pred = np.zeros([all_predictions.shape[1], all_predictions.shape[2]])
    for i, w in enumerate(weights):
        pred += all_predictions[i] * w

    return pred


def main(predictions_dir, basename, task_type, metric, limit, output_dir):
    index_run = 0
    current_num_models = 0
    watch = util.Stopwatch.StopWatch()
    watch.start_task("ensemble_builder")
    used_time = 0
    while used_time < limit:
        #=== Load the dataset information
        info = getInfoFromFile(data_dir, basename)
        print info

        #=== Load the true labels of the validation data
        true_labels = np.load(os.path.join(predictions_dir, "true_labels_ensemble.npy"))

        #=== Load the predictions from the models
        all_predictions_train = []
        dir_ensemble = os.path.join(predictions_dir, "predictions_ensemble/")
        dir_valid = os.path.join(predictions_dir, "predictions_valid/")
        if not os.path.isdir(dir_ensemble):
            # prediction directory does not exist
            time.sleep(2)
            continue

        dir_ensemble_list = os.listdir(dir_ensemble)
        dir_valid_list = os.listdir(dir_valid)

        if len(dir_ensemble_list) != len(dir_valid_list):
            # Directories are inconsistent
            time.sleep(2)
            continue

        if len(dir_ensemble_list) <= current_num_models:
            # Nothing has changed since the last time
            time.sleep(2)
            continue

        for f in dir_ensemble_list:
            predictions = np.load(os.path.join(dir_ensemble, f))
            all_predictions_train.append(predictions)

        #=== Compute the weights for the ensemble
        weights = weighted_ensemble(np.array(all_predictions_train),
                                    true_labels, info)

        all_predictions_valid = []
        for f in dir_valid_list:
            predictions = np.load(os.path.join(dir_valid, f))
            all_predictions_valid.append(predictions)

        #=== Compute the ensemble predictions for the valid data
        Y_valid = ensemble_prediction(np.array(all_predictions_valid), weights)

        filename_test = os.path.join(output_dir, basename + '_valid_' + str(index_run).zfill(3) + '.predict')
        data_io.write(os.path.join(predictions_dir, filename_test), Y_valid)

        all_predictions_test = []
        dir_test = os.path.join(predictions_dir, "predictions_test/")
        for f in os.listdir(dir_test):
            predictions = np.load(os.path.join(dir_test, f))
            all_predictions_test.append(predictions)

        #=== Compute the ensemble predictions for the test data
        Y_test = ensemble_prediction(np.array(all_predictions_test), weights)

        filename_test = os.path.join(output_dir, basename + '_test_' + str(index_run).zfill(3) + '.predict')
        data_io.write(os.path.join(predictions_dir, filename_test), Y_test)
        index_run += 1
        used_time = watch.wall_elapsed("ensemble_builder")

        current_num_models = len(dir_ensemble_list)

if __name__ == "__main__":
    main(predictions_dir=sys.argv[1], basename=sys.argv[2],
         task_type=sys.argv[3], metric=sys.argv[4], limit=float(sys.argv[5]),
         output_dir=sys.argv[6])

