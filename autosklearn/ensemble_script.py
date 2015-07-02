'''
Created on Dec 19, 2014

@author: Aaron Klein
'''

import os
import sys
import cma
import time

import logging
import numpy as np

from autosklearn.data import util as data_util
from autosklearn.models import evaluator
import autosklearn.util.stopwatch


def weighted_ensemble_error(weights, *args):
    predictions = args[0]
    true_labels = args[1]
    metric = args[2]
    task_type = args[3]

    weight_prime = weights / weights.sum()
    weighted_predictions = ensemble_prediction(predictions, weight_prime)

    score = evaluator.calculate_score(true_labels, weighted_predictions,
                                     task_type, metric,
                                     weighted_predictions.shape[1])
    return 1 - score


def weighted_ensemble(predictions, true_labels, task_type, metric, weights, tolfun=1e-3):
    seed = np.random.randint(0, 1000)
    logging.debug("CMA-ES uses seed: " + str(seed))
    n_models = predictions.shape[0]
    if n_models > 1:
        res = cma.fmin(weighted_ensemble_error, weights, sigma0=0.25, restarts=1,
                       args=(predictions, true_labels, metric,
                             task_type), options={'bounds': [0, 1],
                                                  'seed': seed,
                                                  'verb_log': 0, # No output files
                                                  'tolfun': tolfun # Default was 1e-11
                                                  })
        weights = np.array(res[0])
    else:
        # Python CMA-ES does not work in a 1-D space
        weights = np.ones([1])
    weights = weights / weights.sum()

    return weights


def ensemble_prediction(all_predictions, weights):
    pred = np.zeros([all_predictions.shape[1], all_predictions.shape[2]])
    for i, w in enumerate(weights):
        pred += all_predictions[i] * w

    return pred


def main(predictions_dir, basename, task_type, metric, limit, output_dir, ensemble_size=None):
    watch = autosklearn.util.stopwatch.StopWatch()
    watch.start_task("ensemble_builder")

    used_time = 0
    time_iter = 0
    index_run = 0
    current_num_models = 0
    logging.basicConfig(filename=os.path.join(predictions_dir, "ensemble.log"), level=logging.DEBUG)

    while used_time < limit:
        logging.debug("Time left: %f" % (limit - used_time))
        logging.debug("Time last iteration: %f" % time_iter)
        # Load the true labels of the validation data
        true_labels = np.load(os.path.join(predictions_dir, "true_labels_ensemble.npy"))

        # Load the predictions from the models
        all_predictions_train = []
        dir_ensemble = os.path.join(predictions_dir, "predictions_ensemble/")
        dir_valid = os.path.join(predictions_dir, "predictions_valid/")
        dir_test = os.path.join(predictions_dir, "predictions_test/")

        if not os.path.isdir(dir_ensemble) or not os.path.isdir(dir_valid) or \
                not os.path.isdir(dir_test):
            logging.debug("Prediction directory does not exist")
            time.sleep(2)
            used_time = watch.wall_elapsed("ensemble_builder")
            continue

        dir_ensemble_list = sorted(os.listdir(dir_ensemble))
        dir_valid_list = sorted(os.listdir(dir_valid))
        dir_test_list = sorted(os.listdir(dir_test))

        if len(dir_ensemble_list) == 0:
            logging.debug("Directories are empty")
            time.sleep(2)
            used_time = watch.wall_elapsed("ensemble_builder")
            continue

        if len(dir_ensemble_list) != len(dir_valid_list):
            logging.debug("Directories are inconsistent")
            time.sleep(2)
            used_time = watch.wall_elapsed("ensemble_builder")
            continue

        if len(dir_ensemble_list) != len(dir_test_list):
            logging.debug("Directories are inconsistent")
            time.sleep(2)
            used_time = watch.wall_elapsed("ensemble_builder")
            continue

        if len(dir_ensemble_list) <= current_num_models:
            logging.debug("Nothing has changed since the last time")
            time.sleep(2)
            used_time = watch.wall_elapsed("ensemble_builder")
            continue

        watch.start_task("ensemble_iter_" + str(index_run))

        # Binary mask where True indicates that the corresponding will be excluded from the ensemble
        exclude_mask = []
        if ensemble_size is not None:
            # Keeps track of the single scores of each model in our ensemble
            scores_nbest = []
            # The indices of the model that are currently in our ensemble
            indices_nbest = []

        model_idx = 0
        for f in dir_ensemble_list:
            predictions = np.load(os.path.join(dir_ensemble, f))
            score = evaluator.calculate_score(true_labels, predictions,
                task_type, metric, predictions.shape[1])

            if ensemble_size is not None:
                if score <= 0.001:
                    exclude_mask.append(True)
                    logging.error("Model only predicts at random: " + f + " has score: " + str(score))
                # If we have less model in our ensemble than ensemble_size add the current model if it is better than random
                elif len(scores_nbest) < ensemble_size:
                    scores_nbest.append(score)
                    indices_nbest.append(model_idx)
                    exclude_mask.append(False)
                else:
                    # Take the worst performing model in our ensemble so far
                    idx = np.argmin(np.array([scores_nbest]))

                    # If the current model is better than the worst model in our ensemble replace it by the current model
                    if(scores_nbest[idx] < score):
                        logging.debug("Worst model in our ensemble: %d with score %f will be replaced by model %d with score %f" % (idx, scores_nbest[idx], model_idx, score))
                        scores_nbest[idx] = score
                        # Exclude the old model
                        exclude_mask[int(indices_nbest[idx])] = True
                        indices_nbest[idx] = model_idx
                        exclude_mask.append(False)
                    # Otherwise exclude the current model from the ensemble
                    else:
                        exclude_mask.append(True)

            else:
                # Load all predictions that are better than random
                if score <= 0.001:
                    exclude_mask.append(True)
                    logging.error("Model only predicts at random: " + f + " has score: " + str(score))
                else:
                    exclude_mask.append(False)
                    all_predictions_train.append(predictions)

            model_idx += 1
            print exclude_mask

        all_predictions_valid = []
        for i, f in enumerate(dir_valid_list):
            predictions = np.load(os.path.join(dir_valid, f))
            if not exclude_mask[i]:
                all_predictions_valid.append(predictions)

        all_predictions_test = []
        for i, f in enumerate(dir_test_list):
            predictions = np.load(os.path.join(dir_test, f))
            if not exclude_mask[i]:
                all_predictions_test.append(predictions)

        if len(all_predictions_train) == len(all_predictions_test) == len(all_predictions_valid) == 0:
            logging.error("All models do just random guessing")
            time.sleep(2)
            continue

        if len(all_predictions_train) == 1:
            logging.debug("Only one model so far we just copy its predictions")
            Y_valid = all_predictions_valid[0]
            Y_test = all_predictions_test[0]
        else:
            try:
                # Compute the weights for the ensemble
                # Use equally initialized weights
                n_models = len(all_predictions_train)
                init_weights = np.ones([n_models]) / n_models

                weights = weighted_ensemble(np.array(all_predictions_train),
                                    true_labels, task_type, metric, init_weights)
            except (ValueError):
                logging.error("Caught ValueError!")
                used_time = watch.wall_elapsed("ensemble_builder")
                continue
            except:
                logging.error("Caught error!")
                used_time = watch.wall_elapsed("ensemble_builder")
                continue

            # Compute the ensemble predictions for the valid data
            Y_valid = ensemble_prediction(np.array(all_predictions_valid), weights)

            # Compute the ensemble predictions for the test data
            Y_test = ensemble_prediction(np.array(all_predictions_test), weights)

        # Save predictions for valid and test data set
        filename_test = os.path.join(output_dir, basename + '_valid_' + str(index_run).zfill(3) + '.predict')
        data_util.save_predictions(os.path.join(predictions_dir, filename_test), Y_valid)

        filename_test = os.path.join(output_dir, basename + '_test_' + str(index_run).zfill(3) + '.predict')
        data_util.save_predictions(os.path.join(predictions_dir, filename_test), Y_test)

        current_num_models = len(dir_ensemble_list)
        watch.stop_task("ensemble_iter_" + str(index_run))
        time_iter = watch.get_wall_dur("ensemble_iter_" + str(index_run))
        used_time = watch.wall_elapsed("ensemble_builder")
        index_run += 1
    return


if __name__ == "__main__":
    main(predictions_dir=sys.argv[1], basename=sys.argv[2],
         task_type=sys.argv[3], metric=sys.argv[4], limit=float(sys.argv[5]),
         output_dir=sys.argv[6], ensemble_size=int(sys.argv[7]))
    sys.exit(0)
