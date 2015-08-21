# -*- encoding: utf-8 -*-
from os.path import join

try:
    from __init__ import *
except ImportError:
    pass

import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np

import six.moves.cPickle as pickle
from autosklearn.constants import STRING_TO_TASK_TYPES
from autosklearn.util.data import save_predictions
from autosklearn.models import evaluator
from autosklearn.util import StopWatch, get_logger, add_file_handler


def build_ensemble(logger, predictions_train, predictions_valid,
                   predictions_test,
                   true_labels, ensemble_size, task_type, metric):
    indices, trajectory = ensemble_selection(predictions_train, true_labels,
                                             ensemble_size, task_type, metric)
    ensemble_predictions_valid = np.mean(
        predictions_valid[indices.astype(int)],
        axis=0)
    ensemble_predictions_test = np.mean(predictions_test[indices.astype(int)],
                                        axis=0)

    logger.info('Trajectory and indices!')
    logger.info(trajectory)
    logger.info(indices)

    return ensemble_predictions_valid, ensemble_predictions_test, \
        trajectory[-1], indices


def pruning(predictions, labels, n_best, task_type, metric):
    perf = np.zeros([predictions.shape[0]])
    for i, p in enumerate(predictions):
        perf[i] = evaluator.calculate_score(labels, predictions, task_type,
                                            metric, predictions.shape[1])

    indcies = np.argsort(perf)[perf.shape[0] - n_best:]
    return indcies


def get_predictions(dir_path, dir_path_list, include_num_runs, re_num_run):
    result = []
    for i, model_name in enumerate(dir_path_list):
        num_run = int(re_num_run.search(model_name).group(1))
        if num_run in include_num_runs:
            predictions = np.load(os.path.join(dir_path, model_name))
            result.append(predictions)
    return result

def original_ensemble_selection(predictions, labels, ensemble_size, task_type,
                                metric,
                                do_pruning=False):
    """Rich Caruana's ensemble selection method."""

    ensemble = []
    trajectory = []
    order = []

    if do_pruning:
        n_best = 20
        indices = pruning(predictions, labels, n_best, task_type, metric)
        for idx in indices:
            ensemble.append(predictions[idx])
            order.append(idx)
            ensemble_ = np.array(ensemble).mean(axis=0)
            ensemble_performance = evaluator.calculate_score(
                labels, ensemble_, task_type, metric, ensemble_.shape[1])
            trajectory.append(ensemble_performance)
        ensemble_size -= n_best

    for i in range(ensemble_size):
        scores = np.zeros([predictions.shape[0]])
        for j, pred in enumerate(predictions):
            ensemble.append(pred)
            ensemble_prediction = np.mean(np.array(ensemble), axis=0)
            scores[j] = evaluator.calculate_score(labels, ensemble_prediction,
                                                  task_type, metric,
                                                  ensemble_prediction.shape[1])
            ensemble.pop()
        best = np.nanargmax(scores)
        ensemble.append(predictions[best])
        trajectory.append(scores[best])
        order.append(best)

    return np.array(order), np.array(trajectory)


def ensemble_selection(predictions, labels, ensemble_size, task_type, metric,
                       do_pruning=False):
    """Fast version of Rich Caruana's ensemble selection method."""

    ensemble = []
    trajectory = []
    order = []

    if do_pruning:
        n_best = 20
        indices = pruning(predictions, labels, n_best, task_type, metric)
        for idx in indices:
            ensemble.append(predictions[idx])
            order.append(idx)
            ensemble_ = np.array(ensemble).mean(axis=0)
            ensemble_performance = evaluator.calculate_score(
                labels, ensemble_, task_type, metric, ensemble_.shape[1])
            trajectory.append(ensemble_performance)
        ensemble_size -= n_best

    for i in range(ensemble_size):
        scores = np.zeros([predictions.shape[0]])
        s = len(ensemble)
        if s == 0:
            weighted_ensemble_prediction = np.zeros(predictions[0].shape)
        else:
            ensemble_prediction = np.mean(np.array(ensemble), axis=0)
            weighted_ensemble_prediction = (s / float(s + 1)
                                            ) * ensemble_prediction
        for j, pred in enumerate(predictions):
            # ensemble.append(pred)
            # ensemble_prediction = np.mean(np.array(ensemble), axis=0)
            fant_ensemble_prediction = weighted_ensemble_prediction + (
                1. / float(s + 1)) * pred

            scores[j] = evaluator.calculate_score(
                labels, fant_ensemble_prediction, task_type, metric,
                fant_ensemble_prediction.shape[1])
            # ensemble.pop()
        best = np.nanargmax(scores)
        ensemble.append(predictions[best])
        trajectory.append(scores[best])
        order.append(best)

    return np.array(order), np.array(trajectory)


def ensemble_selection_bagging(predictions, labels, ensemble_size, task_type,
                               metric,
                               fraction=0.5,
                               n_bags=20,
                               do_pruning=False):
    """Rich Caruana's ensemble selection method with bagging."""
    n_models = predictions.shape[0]
    bag_size = int(n_models * fraction)

    order_of_each_bag = []
    for j in range(n_bags):
        # Bagging a set of models
        indices = sorted(random.sample(range(0, n_models), bag_size))
        bag = predictions[indices, :, :]
        order, _ = ensemble_selection(bag, labels, ensemble_size, task_type,
                                      metric, do_pruning)
        order_of_each_bag.append(order)

    return np.array(order_of_each_bag)


def check_data(logger, len_ensemble_list, current_num_models):
    for predict, exp in [
        (len_ensemble_list == 0, 'Directories are empty'),
        (len_ensemble_list <= current_num_models,
         'Nothing has changed since the last time'),
    ]:
        if predict:
            logger.debug(exp)
            return True
    return False



def main(logger,
         predictions_dir,
         basename,
         task_type,
         metric,
         limit,
         output_dir,
         ensemble_size=None,
         seed=1,
         indices_output_dir='.'):

    watch = StopWatch()
    watch.start_task('ensemble_builder')

    task_type = STRING_TO_TASK_TYPES[task_type]

    used_time = 0
    time_iter = 0
    index_run = 0
    current_num_models = 0

    dir_ensemble = join(predictions_dir, 'predictions_ensemble_%s/' % seed)
    dir_valid = join(predictions_dir, 'predictions_valid_%s/' % seed)
    dir_test = join(predictions_dir, 'predictions_test_%s/' % seed)
    paths_ = [dir_ensemble, dir_valid, dir_test]

    tru_labels_path = join(predictions_dir, 'true_labels_ensemble.npy')

    while used_time < limit:
        logger.debug('Time left: %f', limit - used_time)
        logger.debug('Time last iteration: %f', time_iter)
        # Load the true labels of the validation data
        true_labels = np.load(tru_labels_path)

        # Load the predictions from the models
        exists = [os.path.isdir(dir_) for dir_ in paths_]
        if not exists[0]:  # all(exists):
            logger.debug('Prediction directory %s does not exist!' %
                          dir_ensemble)
            time.sleep(2)
            used_time = watch.wall_elapsed('ensemble_builder')
            continue

        dir_ensemble_list = sorted(os.listdir(dir_ensemble))
        dir_valid_list = sorted(os.listdir(dir_valid)) if exists[1] else []
        dir_test_list = sorted(os.listdir(dir_test)) if exists[2] else []

        if check_data(logger, len(dir_ensemble_list), current_num_models):
            time.sleep(2)
            used_time = watch.wall_elapsed('ensemble_builder')
            continue

        watch.start_task('ensemble_iter_' + str(index_run))

        # List of num_runs (which are in the filename) which will be included
        #  later
        include_num_runs = []
        re_num_run = re.compile(r'_([0-9]*)\.npy$')
        if ensemble_size is not None:
            # Keeps track of the single scores of each model in our ensemble
            scores_nbest = []
            # The indices of the model that are currently in our ensemble
            indices_nbest = []
            # The names of the models
            model_names = []
            # The num run of the models
            num_runs = []

        model_names_to_scores = dict()

        model_idx = 0
        for model_name in dir_ensemble_list:
            predictions = np.load(os.path.join(dir_ensemble, model_name))
            score = evaluator.calculate_score(true_labels, predictions,
                                              task_type, metric,
                                              predictions.shape[1])
            model_names_to_scores[model_name] = score
            num_run = int(re_num_run.search(model_name).group(1))

            if ensemble_size is not None:
                if score <= 0.001:
                    # include_num_runs.append(True)
                    logger.error('Model only predicts at random: ' +
                                  model_name + ' has score: ' + str(score))
                # If we have less models in our ensemble than ensemble_size add
                # the current model if it is better than random
                elif len(scores_nbest) < ensemble_size:
                    scores_nbest.append(score)
                    indices_nbest.append(model_idx)
                    include_num_runs.append(num_run)
                    model_names.append(model_name)
                    num_runs.append(num_run)
                else:
                    # Take the worst performing model in our ensemble so far
                    idx = np.argmin(np.array([scores_nbest]))

                    # If the current model is better than the worst model in
                    # our ensemble replace it by the current model
                    if scores_nbest[idx] < score:
                        logger.debug('Worst model in our ensemble: %s with '
                                      'score %f will be replaced by model %s '
                                      'with score %f', model_names[idx],
                                      scores_nbest[idx], model_name, score)
                        # Exclude the old model
                        del scores_nbest[idx]
                        scores_nbest.append(score)
                        del include_num_runs[idx]
                        del indices_nbest[idx]
                        indices_nbest.append(model_idx)
                        include_num_runs.append(num_run)
                        del model_names[idx]
                        model_names.append(model_name)
                        del num_runs[idx]
                        num_runs.append(num_run)

                    # Otherwise exclude the current model from the ensemble
                    else:
                        # include_num_runs.append(True)
                        pass

            else:
                # Load all predictions that are better than random
                if score <= 0.001:
                    # include_num_runs.append(True)
                    logger.error('Model only predicts at random: ' +
                                  model_name + ' has score: ' + str(score))
                else:
                    include_num_runs.append(num_run)

            model_idx += 1

        indices_to_model_names = dict()
        indices_to_run_num = dict()
        for i, model_name in enumerate(dir_ensemble_list):
            num_run = int(re_num_run.search(model_name).group(1))
            if num_run in include_num_runs:
                num_indices = len(indices_to_model_names)
                indices_to_model_names[num_indices] = model_name
                indices_to_run_num[num_indices] = num_run

        # logging.info("Indices to model names:")
        # logging.info(indices_to_model_names)

        # for i, item in enumerate(sorted(model_names_to_scores.items(),
        #                                key=lambda t: t[1])):
        #    logging.info("%d: %s", i, item)

        include_num_runs = set(include_num_runs)

        all_predictions_train = get_predictions(dir_ensemble,
                                                dir_ensemble_list,
                                                include_num_runs,
                                                re_num_run)
        all_predictions_valid = get_predictions(dir_valid,
                                                dir_valid_list,
                                                include_num_runs,
                                                re_num_run)

        all_predictions_test = get_predictions(dir_test,
                                               dir_test_list,
                                               include_num_runs,
                                               re_num_run)

        if len(all_predictions_train) == len(all_predictions_test) == len(
                all_predictions_valid) == 0:
            logger.error('All models do just random guessing')
            time.sleep(2)
            continue

        elif len(all_predictions_train) == 1:
            logger.debug('Only one model so far we just copy its predictions')
            ensemble_members_run_numbers = {0: 1.0}

            # Output the score
            logger.info('Training performance: %f' %
                         np.max(model_names_to_scores.values()))
        else:
            try:
                indices, trajectory = ensemble_selection(
                    np.array(all_predictions_train), true_labels,
                    ensemble_size, task_type, metric)

                logger.info('Trajectory and indices!')
                logger.info(trajectory)
                logger.info(indices)

            except ValueError as e:
                logger.error('Caught ValueError: ' + str(e))
                used_time = watch.wall_elapsed('ensemble_builder')
                continue
            except Exception as e:
                logger.error('Caught error! %s', e.message)
                used_time = watch.wall_elapsed('ensemble_builder')
                continue

            # Output the score
            logger.info('Training performance: %f' % trajectory[-1])

            # Print the ensemble members:
            ensemble_members_run_numbers = dict()
            ensemble_members = Counter(indices).most_common()
            ensemble_members_string = 'Ensemble members:\n'
            logger.info(ensemble_members)
            for ensemble_member in ensemble_members:
                weight = float(ensemble_member[1]) / len(indices)
                ensemble_members_string += \
                    ('    %s; weight: %10f; performance: %10f\n' %
                     (indices_to_model_names[ensemble_member[0]],
                      weight,
                      model_names_to_scores[
                         indices_to_model_names[ensemble_member[0]]]))

                ensemble_members_run_numbers[
                    indices_to_run_num[
                        ensemble_member[0]]] = weight
            logger.info(ensemble_members_string)

        # Save the ensemble indices for later use!
        filename_indices = os.path.join(indices_output_dir,
                                        str(index_run).zfill(5) + '.indices')

        logger.info(ensemble_members_run_numbers)
        with open(filename_indices, 'w') as fh:
            pickle.dump(ensemble_members_run_numbers, fh)

        # Save predictions for valid and test data set
        if len(dir_valid_list) == len(dir_ensemble_list):
            ensemble_predictions_valid = np.mean(
                all_predictions_valid[indices.astype(int)],
                axis=0)
            filename_test = os.path.join(
                output_dir,
                basename + '_valid_' + str(index_run).zfill(3) + '.predict')
            save_predictions(
                os.path.join(predictions_dir, filename_test),
                ensemble_predictions_valid)
        else:
            logger.info('Could not find as many validation set predictions '
                         'as ensemble predictions!.')

        if len(dir_test_list) == len(dir_ensemble_list):
            ensemble_predictions_test = np.mean(
                all_predictions_test[indices.astype(int)],
                axis=0)
            filename_test = os.path.join(
                output_dir,
                basename + '_test_' + str(index_run).zfill(3) + '.predict')
            save_predictions(
                os.path.join(predictions_dir, filename_test),
                ensemble_predictions_test)
        else:
            logger.info('Could not find as many test set predictions as '
                         'ensemble predictions!')

        current_num_models = len(dir_ensemble_list)
        watch.stop_task('ensemble_iter_' + str(index_run))
        time_iter = watch.get_wall_dur('ensemble_iter_' + str(index_run))
        used_time = watch.wall_elapsed('ensemble_builder')
        index_run += 1
    return


if __name__ == '__main__':
    seed = int(sys.argv[8])
    predictions_dir = sys.argv[1]

    logger = get_logger(os.path.basename(__file__))
    add_file_handler(logger, os.path.join(predictions_dir, 'ensemble_%d.log' % seed))
    logger.debug("Start script: %s" % __file__)

    main(predictions_dir=sys.argv[1],
         basename=sys.argv[2],
         task_type=sys.argv[3],
         metric=sys.argv[4],
         limit=float(sys.argv[5]),
         output_dir=sys.argv[6],
         ensemble_size=int(sys.argv[7]),
         seed=int(sys.argv[8]),
         indices_output_dir=sys.argv[9],
         logger=logger)
    sys.exit(0)
