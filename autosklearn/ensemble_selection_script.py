'''
Created on Apr 2, 2015

@author: Aaron Klein
'''


from collections import Counter
import os
import sys
import time
import random
import logging
import numpy as np

from autosklearn.data import data_io
from autosklearn.models import evaluator
from autosklearn.util.stopwatch import StopWatch


def build_ensemble(predictions_train, predictions_valid, predictions_test, true_labels, ensemble_size, task_type, metric):

    indices, trajectory = ensemble_selection(predictions_train, true_labels, ensemble_size, task_type, metric)
    ensemble_predictions_valid = np.mean(predictions_valid[indices.astype(int)], axis=0)
    ensemble_predictions_test = np.mean(predictions_test[indices.astype(int)], axis=0)

    return ensemble_predictions_valid, ensemble_predictions_test, \
           trajectory[-1], indices


def pruning(predictions, labels, n_best, task_type, metric):
    perf = np.zeros([predictions.shape[0]])
    for i, p in enumerate(predictions):
        perf[i] = evaluator.calculate_score(labels, predictions, task_type, metric)

    indcies = np.argsort(perf)[perf.shape[0] - n_best:]
    return indcies


def ensemble_selection(predictions, labels, ensemble_size, task_type, metric, do_pruning=False):
    '''
        Rich Caruana's ensemble selection method
    '''

    ensemble = []
    trajectory = []
    order = []

    if do_pruning:
        n_best = 20
        indices = pruning(predictions, labels, n_best, task_type, metric)
        for idx in indices:
            ensemble.append(predictions[idx])
            order.append(idx)
        ensemble_performance = evaluator.calculate_score(labels, np.array(ensemble).mean(axis=0), task_type, metric)
        trajectory.append(ensemble_performance)

    for i in range(ensemble_size):
        scores = np.zeros([predictions.shape[0]])
        for j, pred in enumerate(predictions):
            ensemble.append(pred)
            ensemble_prediction = np.mean(np.array(ensemble), axis=0)
            scores[j] = evaluator.calculate_score(labels, ensemble_prediction, task_type, metric)
            ensemble.pop()
        best = np.argmax(scores)

        ensemble.append(predictions[best])
        trajectory.append(scores[best])
        order.append(best)

    return np.array(order), np.array(trajectory)


def ensemble_selection_bagging(predictions, labels, ensemble_size, task_type, metric, fraction=0.5, n_bags=20, do_pruning=False):
    '''
        Rich Caruana's ensemble selection method with bagging
    '''
    n_models = predictions.shape[0]
    bag_size = int(n_models * fraction)

    order_of_each_bag = []
    for j in range(n_bags):
        # Bagging a set of models
        indices = sorted(random.sample(range(0, n_models), bag_size))
        bag = predictions[indices, :, :]
        order, _ = ensemble_selection(bag, labels, ensemble_size, task_type, metric, do_pruning)
        order_of_each_bag.append(order)

    return np.array(order_of_each_bag)


def main(predictions_dir, basename, task_type, metric, limit, output_dir, ensemble_size=None):
    watch = StopWatch()
    watch.start_task("ensemble_builder")

    used_time = 0
    time_iter = 0
    index_run = 0
    current_num_models = 0
    logging.basicConfig(filename=os.path.join(predictions_dir, "ensemble.log"), level=logging.DEBUG)

    while used_time < limit:
        logging.debug("Time left: %f", limit - used_time)
        logging.debug("Time last iteration: %f", time_iter)
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
        for model_name in dir_ensemble_list:
            predictions = np.load(os.path.join(dir_ensemble, model_name))
            score = evaluator.calculate_score(true_labels, predictions,
                                     task_type, metric)

            if ensemble_size is not None:
                if score <= 0.001:
                    exclude_mask.append(True)
                    logging.error("Model only predicts at random: " + model_name + " has score: " + str(score))
                # If we have less models in our ensemble than ensemble_size add the current model if it is better than random
                elif len(scores_nbest) < ensemble_size:
                    scores_nbest.append(score)
                    indices_nbest.append(model_idx)
                    exclude_mask.append(False)
                    all_predictions_train.append(predictions)
                else:
                    # Take the worst performing model in our ensemble so far
                    idx = np.argmin(np.array([scores_nbest]))

                    # If the current model is better than the worst model in our ensemble replace it by the current model
                    if(scores_nbest[idx] < score):
                        logging.debug("Worst model in our ensemble: %d with score %f will be replaced by model %d with score %f",
                                      idx, scores_nbest[idx], model_idx, score)
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
                    logging.error("Model only predicts at random: " + model_name + " has score: " + str(score))
                else:
                    exclude_mask.append(False)
                    all_predictions_train.append(predictions)

            model_idx += 1

        indices_to_model_names = dict()
        all_predictions_valid = []

        for i, model_name in enumerate(dir_ensemble_list):
            if not exclude_mask[i]:
                num_indices = len(indices_to_model_names)
                indices_to_model_names[num_indices] = model_name

        for i, model_name in enumerate(dir_valid_list):
            predictions = np.load(os.path.join(dir_valid, model_name))
            if not exclude_mask[i]:
                all_predictions_valid.append(predictions)

        all_predictions_test = []
        for i, model_name in enumerate(dir_test_list):
            predictions = np.load(os.path.join(dir_test, model_name))
            if not exclude_mask[i]:
                all_predictions_test.append(predictions)

        if len(all_predictions_train) == len(all_predictions_test) == len(all_predictions_valid) == 0:
            logging.error("All models do just random guessing")
            time.sleep(2)
            continue

        elif len(all_predictions_train) == 1:
            logging.debug("Only one model so far we just copy its predictions")
            Y_valid = all_predictions_valid[0]
            Y_test = all_predictions_test[0]

            # Output the score
            logging.info("Training performance: %f" % score)
        else:
            try:
                Y_valid, Y_test, score, indices = build_ensemble(
                    np.array(all_predictions_train),
                    np.array(all_predictions_valid),
                    np.array(all_predictions_test),
                    true_labels, ensemble_size, task_type, metric)
            except (ValueError):
                logging.error("Caught ValueError!")
                used_time = watch.wall_elapsed("ensemble_builder")
                continue
            except Exception as e:
                logging.error("Caught error! %s", e.message)
                used_time = watch.wall_elapsed("ensemble_builder")
                continue

            # Output the score
            logging.info("Training performance: %f" % score)

            # Print the ensemble members:
            ensemble_members = Counter(indices).most_common()
            ensemble_members_string = "Ensemble members:\n"
            logging.info(ensemble_members)
            for ensemble_member in ensemble_members:
                ensemble_members_string += \
                    ("    %s; weight: %f\n" %
                     (indices_to_model_names[ensemble_member[0]],
                      float(ensemble_member[1]) / len(indices)))
            logging.info(ensemble_members_string)

        # Save predictions for valid and test data set
        filename_test = os.path.join(output_dir, basename + '_valid_' + str(index_run).zfill(3) + '.predict')
        data_io.write(os.path.join(predictions_dir, filename_test), Y_valid)

        filename_test = os.path.join(output_dir, basename + '_test_' + str(index_run).zfill(3) + '.predict')
        data_io.write(os.path.join(predictions_dir, filename_test), Y_test)

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
