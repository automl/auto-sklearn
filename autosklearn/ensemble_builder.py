# -*- encoding: utf-8 -*-

import multiprocessing
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import pynisher

from autosklearn.constants import STRING_TO_TASK_TYPES, STRING_TO_METRIC, \
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, \
    MULTILABEL_CLASSIFICATION, CLASSIFICATION_TASKS, REGRESSION_TASKS, \
    BAC_METRIC, F1_METRIC
from autosklearn.evaluation.util import calculate_score
from autosklearn.util import StopWatch, Backend
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.util.logging_ import get_logger, setup_logger


class EnsembleBuilder(multiprocessing.Process):
    def __init__(self, backend, dataset_name, task_type, metric,
                 limit, ensemble_size=None, ensemble_nbest=None,
                 seed=1, shared_mode=False, max_iterations=-1, precision="32",
                 low_precision=True):
        super(EnsembleBuilder, self).__init__()

        self.backend = backend
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.metric = metric
        self.limit = limit
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.shared_mode = shared_mode
        self.max_iterations = max_iterations
        self.precision = precision
        self.low_precision = low_precision

        logger_name = 'EnsembleBuilder(%d):%s' % (self.seed, self.dataset_name)
        self.logger = get_logger(logger_name)

    def run(self):
        buffer_time = 5
        time_left = self.limit - buffer_time
        safe_ensemble_script = pynisher.enforce_limits(
            wall_time_in_s=int(time_left), logger=self.logger)(self.main)
        safe_ensemble_script()

    def main(self):

        watch = StopWatch()
        watch.start_task('ensemble_builder')

        used_time = 0
        time_iter = 0
        index_run = 0
        num_iteration = 0
        current_num_models = 0
        last_hash = None
        current_hash = None

        dir_ensemble = os.path.join(self.backend.temporary_directory,
                                    '.auto-sklearn',
                                    'predictions_ensemble')
        dir_valid = os.path.join(self.backend.temporary_directory,
                                 '.auto-sklearn',
                                 'predictions_valid')
        dir_test = os.path.join(self.backend.temporary_directory,
                                '.auto-sklearn',
                                'predictions_test')
        paths_ = [dir_ensemble, dir_valid, dir_test]

        dir_ensemble_list_mtimes = []

        self.logger.debug('Starting main loop with %f seconds and %d iterations '
                          'left.' % (self.limit - used_time, num_iteration))
        while used_time < self.limit or (self.max_iterations > 0 and
                                         self.max_iterations >= num_iteration):
            num_iteration += 1
            self.logger.debug('Time left: %f', self.limit - used_time)
            self.logger.debug('Time last ensemble building: %f', time_iter)

            # Reload the ensemble targets every iteration, important, because cv may
            # update the ensemble targets in the cause of running auto-sklearn
            # TODO update cv in order to not need this any more!
            targets_ensemble = self.backend.load_targets_ensemble()

            # Load the predictions from the models
            exists = [os.path.isdir(dir_) for dir_ in paths_]
            if not exists[0]:  # all(exists):
                self.logger.debug('Prediction directory %s does not exist!' %
                              dir_ensemble)
                time.sleep(2)
                used_time = watch.wall_elapsed('ensemble_builder')
                continue

            if self.shared_mode is False:
                dir_ensemble_list = sorted(glob.glob(os.path.join(
                    dir_ensemble, 'predictions_ensemble_%s_*.npy' % self.seed)))
                if exists[1]:
                    dir_valid_list = sorted(glob.glob(os.path.join(
                        dir_valid, 'predictions_valid_%s_*.npy' % self.seed)))
                else:
                    dir_valid_list = []
                if exists[2]:
                    dir_test_list = sorted(glob.glob(os.path.join(
                        dir_test, 'predictions_test_%s_*.npy' % self.seed)))
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
            # The ensemble dir can contain non-model files. We filter them and
            # use the following list instead
            dir_ensemble_model_files = []

            for dir_ensemble_file in dir_ensemble_list:
                if dir_ensemble_file.endswith("/"):
                    dir_ensemble_file = dir_ensemble_file[:-1]
                if not dir_ensemble_file.endswith(".npy"):
                    self.logger.warning('Error loading file (not .npy): %s', dir_ensemble_file)
                    continue

                dir_ensemble_model_files.append(dir_ensemble_file)
                basename = os.path.basename(dir_ensemble_file)
                dir_ensemble_file = os.path.join(dir_ensemble, basename)
                mtime = os.path.getmtime(dir_ensemble_file)
                dir_ensemble_list_mtimes.append(mtime)

            if len(dir_ensemble_model_files) == 0:
                self.logger.debug('Directories are empty')
                time.sleep(2)
                used_time = watch.wall_elapsed('ensemble_builder')
                continue

            if len(dir_ensemble_model_files) <= current_num_models and \
                    old_dir_ensemble_list_mtimes == dir_ensemble_list_mtimes:
                self.logger.debug('Nothing has changed since the last time')
                time.sleep(2)
                used_time = watch.wall_elapsed('ensemble_builder')
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # TODO restructure time management in the ensemble builder,
                # what is the time of index_run actually needed for?
                watch.start_task('index_run' + str(index_run))
            watch.start_task('ensemble_iter_' + str(num_iteration))

            # List of num_runs (which are in the filename) which will be included
            #  later
            include_num_runs = []
            backup_num_runs = []
            model_and_automl_re = re.compile(r'_([0-9]*)_([0-9]*)\.npy$')
            if self.ensemble_nbest is not None:
                # Keeps track of the single scores of each model in our ensemble
                scores_nbest = []
                # The indices of the model that are currently in our ensemble
                indices_nbest = []
                # The names of the models
                model_names = []

            model_names_to_scores = dict()

            model_idx = 0
            for model_name in dir_ensemble_model_files:
                if model_name.endswith("/"):
                    model_name = model_name[:-1]
                basename = os.path.basename(model_name)

                try:
                    if self.precision is "16":
                        predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float16)
                    elif self.precision is "32":
                        predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float32)
                    elif self.precision is "64":
                        predictions = np.load(os.path.join(dir_ensemble, basename)).astype(dtype=np.float64)
                    else:
                        predictions = np.load(os.path.join(dir_ensemble, basename))

                    score = calculate_score(targets_ensemble, predictions,
                                            self.task_type, self.metric,
                                            predictions.shape[1])

                except Exception as e:
                    self.logger.warning('Error loading %s: %s - %s',
                                        basename, type(e), e)
                    score = -1

                model_names_to_scores[model_name] = score
                match = model_and_automl_re.search(model_name)
                automl_seed = int(match.group(1))
                num_run = int(match.group(2))

                if self.ensemble_nbest is not None:
                    if score <= 0.001:
                        self.logger.info('Model only predicts at random: ' +
                                         model_name + ' has score: ' + str(score))
                        backup_num_runs.append((automl_seed, num_run))
                    # If we have less models in our ensemble than ensemble_nbest add
                    # the current model if it is better than random
                    elif len(scores_nbest) < self.ensemble_nbest:
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
                            self.logger.info(
                                'Worst model in our ensemble: %s with score %f '
                                'will be replaced by model %s with score %f',
                                model_names[idx], scores_nbest[idx], model_name,
                                score)
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
                        self.logger.info('Model only predicts at random: ' +
                                         model_name + ' has score: ' +
                                         str(score))
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
            for i, model_name in enumerate(dir_ensemble_model_files):
                match = model_and_automl_re.search(model_name)
                automl_seed = int(match.group(1))
                num_run = int(match.group(2))
                if (automl_seed, num_run) in include_num_runs:
                    num_indices = len(indices_to_model_names)
                    indices_to_model_names[num_indices] = model_name
                    indices_to_run_num[num_indices] = (automl_seed, num_run)

            try:
                all_predictions_train, all_predictions_valid, all_predictions_test =\
                    self.get_all_predictions(dir_ensemble,
                                             dir_ensemble_model_files,
                                             dir_valid, dir_valid_list,
                                             dir_test, dir_test_list,
                                             include_num_runs,
                                             model_and_automl_re,
                                             self.precision)
            except IOError:
                self.logger.error('Could not load the predictions.')
                continue

            if len(include_num_runs) == 0:
                self.logger.error('All models do just random guessing')
                time.sleep(2)
                continue

            else:
                ensemble = EnsembleSelection(ensemble_size=self.ensemble_size,
                                             task_type=self.task_type,
                                             metric=self.metric)

                try:
                    ensemble.fit(all_predictions_train, targets_ensemble,
                                 include_num_runs)
                    self.logger.info(ensemble)

                except ValueError as e:
                    self.logger.error('Caught ValueError: ' + str(e))
                    used_time = watch.wall_elapsed('ensemble_builder')
                    time.sleep(2)
                    continue
                except IndexError as e:
                    self.logger.error('Caught IndexError: ' + str(e))
                    used_time = watch.wall_elapsed('ensemble_builder')
                    time.sleep(2)
                    continue
                except Exception as e:
                    self.logger.error('Caught error! %s', str(e))
                    used_time = watch.wall_elapsed('ensemble_builder')
                    time.sleep(2)
                    continue

                # Output the score
                self.logger.info('Training performance: %f' % ensemble.train_score_)

                self.logger.info('Building the ensemble took %f seconds' %
                            watch.wall_elapsed('ensemble_iter_' + str(num_iteration)))

            # Set this variable here to avoid re-running the ensemble builder
            # every two seconds in case the ensemble did not change
            current_num_models = len(dir_ensemble_model_files)

            ensemble_predictions = ensemble.predict(all_predictions_train)
            if sys.version_info[0] == 2:
                ensemble_predictions.flags.writeable = False
                current_hash = hash(ensemble_predictions.data)
            else:
                current_hash = hash(ensemble_predictions.data.tobytes())

            # Only output a new ensemble and new predictions if the output of the
            # ensemble would actually change!
            # TODO this is neither safe (collisions, tests only with the ensemble
            #  prediction, but not the ensemble), implement a hash function for
            # each possible ensemble builder.
            if last_hash is not None:
                if current_hash == last_hash:
                    self.logger.info('Ensemble output did not change.')
                    time.sleep(2)
                    continue
                else:
                    last_hash = current_hash
            else:
                last_hash = current_hash

            # Save the ensemble for later use in the main auto-sklearn module!
            self.backend.save_ensemble(ensemble, index_run, self.seed)

            # Save predictions for valid and test data set
            if len(dir_valid_list) == len(dir_ensemble_model_files):
                all_predictions_valid = np.array(all_predictions_valid)
                ensemble_predictions_valid = ensemble.predict(all_predictions_valid)
                if self.task_type == BINARY_CLASSIFICATION:
                    ensemble_predictions_valid = ensemble_predictions_valid[:, 1]
                if self.low_precision:
                    if self.task_type in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION]:
                        ensemble_predictions_valid[ensemble_predictions_valid < 1e-4] = 0.
                    if self.metric in [BAC_METRIC, F1_METRIC]:
                        bin_array = np.zeros(ensemble_predictions_valid.shape, dtype=np.int32)
                        if (self.task_type != MULTICLASS_CLASSIFICATION) or (
                            ensemble_predictions_valid.shape[1] == 1):
                            bin_array[ensemble_predictions_valid >= 0.5] = 1
                        else:
                            sample_num = ensemble_predictions_valid.shape[0]
                            for i in range(sample_num):
                                j = np.argmax(ensemble_predictions_valid[i, :])
                                bin_array[i, j] = 1
                        ensemble_predictions_valid = bin_array
                    if self.task_type in CLASSIFICATION_TASKS:
                        if ensemble_predictions_valid.size < (20000 * 20):
                            precision = 3
                        else:
                            precision = 2
                    else:
                        if ensemble_predictions_valid.size > 1000000:
                            precision = 4
                        else:
                            # File size maximally 2.1MB
                            precision = 6

                self.backend.save_predictions_as_txt(ensemble_predictions_valid,
                                                'valid', index_run, prefix=self.dataset_name,
                                                precision=precision)
            else:
                self.logger.info('Could not find as many validation set predictions (%d)'
                             'as ensemble predictions (%d)!.',
                            len(dir_valid_list), len(dir_ensemble_model_files))

            del all_predictions_valid

            if len(dir_test_list) == len(dir_ensemble_model_files):
                all_predictions_test = np.array(all_predictions_test)
                ensemble_predictions_test = ensemble.predict(all_predictions_test)
                if self.task_type == BINARY_CLASSIFICATION:
                    ensemble_predictions_test = ensemble_predictions_test[:, 1]
                if self.low_precision:
                    if self.task_type in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION]:
                        ensemble_predictions_test[ensemble_predictions_test < 1e-4] = 0.
                    if self.metric in [BAC_METRIC, F1_METRIC]:
                        bin_array = np.zeros(ensemble_predictions_test.shape,
                                             dtype=np.int32)
                        if (self.task_type != MULTICLASS_CLASSIFICATION) or (
                                    ensemble_predictions_test.shape[1] == 1):
                            bin_array[ensemble_predictions_test >= 0.5] = 1
                        else:
                            sample_num = ensemble_predictions_test.shape[0]
                            for i in range(sample_num):
                                j = np.argmax(ensemble_predictions_test[i, :])
                                bin_array[i, j] = 1
                        ensemble_predictions_test = bin_array
                    if self.task_type in CLASSIFICATION_TASKS:
                        if ensemble_predictions_test.size < (20000 * 20):
                            precision = 3
                        else:
                            precision = 2
                    else:
                        if ensemble_predictions_test.size > 1000000:
                            precision = 4
                        else:
                            precision = 6

                self.backend.save_predictions_as_txt(ensemble_predictions_test,
                                                     'test', index_run, prefix=self.dataset_name,
                                                     precision=precision)
            else:
                self.logger.info('Could not find as many test set predictions (%d) as '
                             'ensemble predictions (%d)!',
                            len(dir_test_list), len(dir_ensemble_model_files))

            del all_predictions_test

            current_num_models = len(dir_ensemble_model_files)
            watch.stop_task('index_run' + str(index_run))
            time_iter = watch.get_wall_dur('index_run' + str(index_run))
            used_time = watch.wall_elapsed('ensemble_builder')
            index_run += 1
        return

    def get_predictions(self, dir_path, dir_path_list, include_num_runs,
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
                    predictions = np.load(os.path.join(dir_path, basename)).astype(
                        dtype=np.float16)
                elif precision == "32":
                    predictions = np.load(os.path.join(dir_path, basename)).astype(
                        dtype=np.float32)
                elif precision == "64":
                    predictions = np.load(os.path.join(dir_path, basename)).astype(
                        dtype=np.float64)
                else:
                    predictions = np.load(os.path.join(dir_path, basename))
                result.append(predictions)
        return result


    def get_all_predictions(self, dir_train, dir_train_list,
                            dir_valid, dir_valid_list,
                            dir_test, dir_test_list,
                            include_num_runs,
                            model_and_automl_re, precision="32"):
        train_pred = self.get_predictions(dir_train, dir_train_list,
                                          include_num_runs,
                                          model_and_automl_re, precision)
        valid_pred = self.get_predictions(dir_valid, dir_valid_list,
                                          include_num_runs,
                                          model_and_automl_re, precision)
        test_pred = self.get_predictions(dir_test, dir_test_list,
                                         include_num_runs,
                                         model_and_automl_re, precision)
        return train_pred, valid_pred, test_pred
