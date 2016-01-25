# -*- encoding: utf-8 -*-

import argparse
import glob
import logging
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np

from autosklearn.constants import STRING_TO_TASK_TYPES, STRING_TO_METRIC
from autosklearn.evaluation.util import calculate_score
from autosklearn.util import StopWatch, Backend


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("ensemble_builder")
logger.setLevel(logging.DEBUG)


def build_ensemble(predictions_train, predictions_valid, predictions_test,
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
        perf[i] = calculate_score(labels, predictions, task_type,
                                  metric, predictions.shape[1])

    indcies = np.argsort(perf)[perf.shape[0] - n_best:]
    return indcies


def get_predictions(dir_path, dir_path_list, include_num_runs,
                    model_and_automl_re, precision="32"):
    result = []
    for i, model_name in enumerate(dir_path_list):
        match = model_and_automl_re.search(model_name)
        automl_seed = int(match.group(1))
        num_run = int(match.group(2))

        if model_name.endswith("/"):
            model_name = model_name[:-1]
        basename = os.path.basename(model_name)

        if (automl_seed, num_run) in include_num_runs:
            if precision == "16":
                predictions = np.load(os.path.join(dir_path, basename)).astype(dtype=np.float16)
            elif precision == "32":
                predictions = np.load(os.path.join(dir_path, basename)).astype(dtype=np.float32)
            elif precision == "64":
                predictions = np.load(os.path.join(dir_path, basename)).astype(dtype=np.float64)
            else:
                predictions = np.load(os.path.join(dir_path, basename))
            result.append(predictions)
    return result


def original_ensemble_selection(predictions, labels, ensemble_size, task_type,
                                metric, do_pruning=False):
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
            ensemble_performance = calculate_score(
                labels, ensemble_, task_type, metric, ensemble_.shape[1])
            trajectory.append(ensemble_performance)
        ensemble_size -= n_best

    for i in range(ensemble_size):
        scores = np.zeros([predictions.shape[0]])
        for j, pred in enumerate(predictions):
            ensemble.append(pred)
            ensemble_prediction = np.mean(np.array(ensemble), axis=0)
            scores[j] = calculate_score(labels, ensemble_prediction,
                                        task_type, metric,
                                        ensemble_prediction.shape[1])
            ensemble.pop()
        best = np.nanargmax(scores)
        ensemble.append(predictions[best])
        trajectory.append(scores[best])
        order.append(best)

        # Handle special case
        if len(predictions) == 1:
            break

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
            ensemble_performance = calculate_score(
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

            scores[j] = calculate_score(
                labels, fant_ensemble_prediction, task_type, metric,
                fant_ensemble_prediction.shape[1])
            # ensemble.pop()
        best = np.nanargmax(scores)
        ensemble.append(predictions[best])
        trajectory.append(scores[best])
        order.append(best)

        # Handle special case
        if len(predictions) == 1:
            break

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


def main(autosklearn_tmp_dir,
         dataset_name,
         task_type,
         metric,
         limit,
         output_dir,
         ensemble_size=None,
         ensemble_nbest=None,
         seed=1,
         shared_mode=False,
         max_iterations=-1,
         precision="32"):

    watch = StopWatch()
    watch.start_task('ensemble_builder')

    used_time = 0
    time_iter = 0
    index_run = 0
    num_iteration = 0
    current_num_models = 0

    backend = Backend(output_dir, autosklearn_tmp_dir)
    dir_ensemble = os.path.join(autosklearn_tmp_dir, '.auto-sklearn',
                                'predictions_ensemble')
    dir_valid = os.path.join(autosklearn_tmp_dir, '.auto-sklearn',
                             'predictions_valid')
    dir_test = os.path.join(autosklearn_tmp_dir, '.auto-sklearn',
                            'predictions_test')
    paths_ = [dir_ensemble, dir_valid, dir_test]

    targets_ensemble = backend.load_targets_ensemble()

    dir_ensemble_list_mtimes = []

    while used_time < limit or (max_iterations > 0 and max_iterations >= num_iteration):
        num_iteration += 1
        logger.debug('Time left: %f', limit - used_time)
        logger.debug('Time last iteration: %f', time_iter)

        # Load the predictions from the models
        exists = [os.path.isdir(dir_) for dir_ in paths_]
        if not exists[0]:  # all(exists):
            logger.debug('Prediction directory %s does not exist!' %
                          dir_ensemble)
            time.sleep(2)
            used_time = watch.wall_elapsed('ensemble_builder')
            continue

        if shared_mode is False:
            dir_ensemble_list = sorted(glob.glob(os.path.join(
                dir_ensemble, 'predictions_ensemble_%s_*.npy' % seed)))
            if exists[1]:
                dir_valid_list = sorted(glob.glob(os.path.join(
                    dir_valid, 'predictions_valid_%s_*.npy' % seed)))
            else:
                dir_valid_list = []
            if exists[2]:
                dir_test_list = sorted(glob.glob(os.path.join(
                    dir_test, 'predictions_test_%s_*.npy' % seed)))
            else:
                dir_test_list = []
        else:
            dir_ensemble_list = sorted(os.listdir(dir_ensemble))
            dir_valid_list = sorted(os.listdir(dir_valid)) if exists[1] else []
            dir_test_list = sorted(os.listdir(dir_test)) if exists[2] else []

        # Check the modification times because predictions can be updated
        # over time!
        old_dir_ensemble_list_mtimes = dir_ensemble_list_mtimes
        dir_ensemble_list_mtimes = []

        for dir_ensemble_file in dir_ensemble_list:
            if dir_ensemble_file.endswith("/"):
                dir_ensemble_file = dir_ensemble_file[:-1]
            basename = os.path.basename(dir_ensemble_file)
            dir_ensemble_file = os.path.join(dir_ensemble, basename)
            mtime = os.path.getmtime(dir_ensemble_file)
            dir_ensemble_list_mtimes.append(mtime)

        if len(dir_ensemble_list) == 0:
            logger.debug('Directories are empty')
            time.sleep(2)
            used_time = watch.wall_elapsed('ensemble_builder')
            continue

        if len(dir_ensemble_list) <= current_num_models and \
                old_dir_ensemble_list_mtimes == dir_ensemble_list_mtimes:
            logger.debug('Nothing has changed since the last time')
            time.sleep(2)
            used_time = watch.wall_elapsed('ensemble_builder')
            continue

        watch.start_task('ensemble_iter_' + str(index_run))

        # List of num_runs (which are in the filename) which will be included
        #  later
        include_num_runs = []
        backup_num_runs = []
        model_and_automl_re = re.compile(r'_([0-9]*)_([0-9]*)\.npy$')
        if ensemble_nbest is not None:
            # Keeps track of the single scores of each model in our ensemble
            scores_nbest = []
            # The indices of the model that are currently in our ensemble
            indices_nbest = []
            # The names of the models
            model_names = []

        model_names_to_scores = dict()

        model_idx = 0
        for model_name in dir_ensemble_list:
            if model_name.endswith("/"):
                model_name = model_name[:-1]
            basename = os.path.basename(model_name)

            if precision is "16":
                predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float16)
            elif precision is "32":
                predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float32)
            elif precision is "64":
                predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float64)
            else:
                predictions = np.load(os.path.join(dir_ensemble, basename))
            score = calculate_score(targets_ensemble, predictions,
                                    task_type, metric,
                                    predictions.shape[1])
            model_names_to_scores[model_name] = score
            match = model_and_automl_re.search(model_name)
            automl_seed = int(match.group(1))
            num_run = int(match.group(2))

            if ensemble_nbest is not None:
                if score <= 0.001:
                    logger.error('Model only predicts at random: ' +
                                  model_name + ' has score: ' + str(score))
                    backup_num_runs.append((automl_seed, num_run))
                # If we have less models in our ensemble than ensemble_nbest add
                # the current model if it is better than random
                elif len(scores_nbest) < ensemble_nbest:
                    scores_nbest.append(score)
                    indices_nbest.append(model_idx)
                    include_num_runs.append((automl_seed, num_run))
                    model_names.append(model_name)
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
                        include_num_runs.append((automl_seed, num_run))
                        del model_names[idx]
                        model_names.append(model_name)

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
                    backup_num_runs.append((automl_seed, num_run))
                else:
                    include_num_runs.append((automl_seed, num_run))

            model_idx += 1

        # If there is no model better than random guessing, we have to use
        # all models which do random guessing
        if len(include_num_runs) == 0:
            include_num_runs = backup_num_runs

        indices_to_model_names = dict()
        indices_to_run_num = dict()
        for i, model_name in enumerate(dir_ensemble_list):
            match = model_and_automl_re.search(model_name)
            automl_seed = int(match.group(1))
            num_run = int(match.group(2))
            if (automl_seed, num_run) in include_num_runs:
                num_indices = len(indices_to_model_names)
                indices_to_model_names[num_indices] = model_name
                indices_to_run_num[num_indices] = (automl_seed, num_run)

        # logging.info("Indices to model names:")
        # logging.info(indices_to_model_names)

        # for i, item in enumerate(sorted(model_names_to_scores.items(),
        #                                key=lambda t: t[1])):
        #    logging.info("%d: %s", i, item)

        include_num_runs = set(include_num_runs)

        all_predictions_train = get_predictions(dir_ensemble,
                                                dir_ensemble_list,
                                                include_num_runs,
                                                model_and_automl_re,
                                                precision)

#        if len(all_predictions_train) == len(all_predictions_test) == len(
#                all_predictions_valid) == 0:
        if len(include_num_runs) == 0:
            logger.error('All models do just random guessing')
            time.sleep(2)
            continue

        else:
            try:
                indices, trajectory = ensemble_selection(
                    np.array(all_predictions_train), targets_ensemble,
                    ensemble_size, task_type, metric)

                logger.info('Trajectory and indices!')
                logger.info(trajectory)
                logger.info(indices)

            except ValueError as e:
                logger.error('Caught ValueError: ' + str(e))
                used_time = watch.wall_elapsed('ensemble_builder')
                time.sleep(2)
                continue
            except IndexError as e:
                logger.error('Caught IndexError: ' + str(e))
                used_time = watch.wall_elapsed('ensemble_builder')
                time.sleep(2)
                continue
            except Exception as e:
                logger.error('Caught error! %s', e.message)
                used_time = watch.wall_elapsed('ensemble_builder')
                time.sleep(2)
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
        backend.save_ensemble_indices_weights(ensemble_members_run_numbers,
                                              index_run, seed)

        all_predictions_valid = get_predictions(dir_valid,
                                                dir_valid_list,
                                                include_num_runs,
                                                model_and_automl_re,
                                                precision)

        # Save predictions for valid and test data set
        if len(dir_valid_list) == len(dir_ensemble_list):
            all_predictions_valid = np.array(all_predictions_valid)
            ensemble_predictions_valid = np.mean(
                all_predictions_valid[indices.astype(int)], axis=0)
            backend.save_predictions_as_txt(ensemble_predictions_valid,
                                            'valid', index_run, prefix=dataset_name)
        else:
            logger.info('Could not find as many validation set predictions (%d)'
                         'as ensemble predictions (%d)!.',
                        len(dir_valid_list), len(dir_ensemble_list))

        del all_predictions_valid
        all_predictions_test = get_predictions(dir_test,
                                               dir_test_list,
                                               include_num_runs,
                                               model_and_automl_re,
                                               precision)

        if len(dir_test_list) == len(dir_ensemble_list):
            all_predictions_test = np.array(all_predictions_test)
            ensemble_predictions_test = np.mean(
                all_predictions_test[indices.astype(int)], axis=0)
            backend.save_predictions_as_txt(ensemble_predictions_test,
                                            'test', index_run, prefix=dataset_name)
        else:
            logger.info('Could not find as many test set predictions (%d) as '
                         'ensemble predictions (%d)!',
                        len(dir_test_list), len(dir_ensemble_list))

        del all_predictions_test

        current_num_models = len(dir_ensemble_list)
        watch.stop_task('ensemble_iter_' + str(index_run))
        time_iter = watch.get_wall_dur('ensemble_iter_' + str(index_run))
        used_time = watch.wall_elapsed('ensemble_builder')
        index_run += 1
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-sklearn-tmp-directory', required=True,
                        help='TMP directory of auto-sklearn. Predictions to '
                             'build the ensemble will be read from here and '
                             'the ensemble indices will be saved here.')
    parser.add_argument('--dataset_name', required=True,
                        help='Name of the dataset. Used to prefix prediction '
                             'output files.')
    parser.add_argument('--task', required=True,
                        choices=list(STRING_TO_TASK_TYPES.keys()))
    parser.add_argument('--metric', required=True,
                        choices=list(STRING_TO_METRIC.keys()))
    parser.add_argument('--limit', required=True, type=float,
                        help='Runtime limit given in seconds.')
    parser.add_argument('--output-directory', required=True,
                        help='Output directory of auto-sklearn. Ensemble '
                             'predictions will be written here.')
    parser.add_argument('--ensemble-size', required=True, type=int)
    parser.add_argument('--ensemble-nbest', required=True, type=int)
    parser.add_argument('--auto-sklearn-seed', required=True, type=int,
                        help='Only work on the output data of a specific '
                             'auto-sklearn run, indicated by the seed. If '
                             'negative, this script will work on the output '
                             'of all available auto-sklearn runs.')
    parser.add_argument('--max-iterations', type=int, default=-1,
                        help='Maximum number of iterations. If -1, run until '
                             'time is up.')
    parser.add_argument('--shared-mode', action='store_true',
                        help='If True, build ensemble with all available '
                             'models. Otherwise, use only models produced by '
                             'a SMAC run with the same seed.')
    parser.add_argument('--precision', required=False, default="32",
                        choices=list(["16", "32", "64"]))

    args = parser.parse_args()
    seed = args.auto_sklearn_seed

    log_file = os.path.join(os.getcwd(), "ensemble.out")
    logger.debug("Start script: %s" % __file__)

    task = STRING_TO_TASK_TYPES[args.task]
    metric = STRING_TO_METRIC[args.metric]
    main(autosklearn_tmp_dir=args.auto_sklearn_tmp_directory,
         dataset_name=args.dataset_name,
         task_type=task,
         metric=metric,
         limit=args.limit,
         output_dir=args.output_directory,
         ensemble_size=args.ensemble_size,
         ensemble_nbest=args.ensemble_nbest,
         seed=args.auto_sklearn_seed,
         shared_mode=args.shared_mode,
         max_iterations=args.max_iterations,
         precision=args.precision)
    sys.exit(0)
