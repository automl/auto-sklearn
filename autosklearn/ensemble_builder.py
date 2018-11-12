# -*- encoding: utf-8 -*-

import multiprocessing
import glob
import os
import re
import time
import traceback
from typing import Optional, Union

import numpy as np
import pynisher
from sklearn.utils.validation import check_random_state

from autosklearn.util.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.metrics import calculate_score, Scorer
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.util.logging_ import get_logger

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2


class EnsembleBuilder(multiprocessing.Process):
    def __init__(
            self,
            backend: Backend,
            dataset_name: str,
            task_type: int,
            metric: Scorer,
            limit: int,
            ensemble_size: int=10,
            ensemble_nbest: int=100,
            seed: int=1,
            shared_mode: bool=False,
            max_iterations: int=None,
            precision: str="32",
            sleep_duration: int=2,
            memory_limit: int=1000,
            read_at_most: int=5,
            random_state: Optional[Union[int, np.random.RandomState]]=None,
    ):
        """
            Constructor

            Parameters
            ----------
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                type of ML task
            metric: str
                name of metric to score predictions
            limit: int
                time limit in sec
            ensemble_size: int
                maximal size of ensemble (passed to autosklearn.ensemble.ensemble_selection)
            ensemble_nbest: int
                consider only the n best prediction (wrt validation predictions)
            seed: int
                random seed
                if set to -1, read files with any seed (e.g., for shared model mode)
            shared_model: bool
                auto-sklearn used shared model mode (aka pSMAC)
            max_iterations: int
                maximal number of iterations to run this script
                (default None --> deactivated)
            precision: ["16","32","64","128"]
                precision of floats to read the predictions
            sleep_duration: int
                duration of sleeping time between two iterations of this script (in sec)
            memory_limit: int
                memory limit in mb
            read_at_most: int
                read at most n new prediction files in each iteration
        """

        super(EnsembleBuilder, self).__init__()

        self.backend = backend  # communication with filesystem
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.metric = metric
        self.time_limit = limit  # time limit
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest  # max number of members that will be used for building the ensemble
        self.seed = seed
        self.shared_mode = shared_mode  # pSMAC?
        self.max_iterations = max_iterations
        self.precision = precision
        self.sleep_duration = sleep_duration
        self.memory_limit = memory_limit
        self.read_at_most = read_at_most
        self.random_state = check_random_state(random_state)

        # part of the original training set
        # used to build the ensemble
        self.dir_ensemble = os.path.join(
            self.backend.temporary_directory,
            '.auto-sklearn',
            'predictions_ensemble',
        )

        # validation set (public test set) -- y_true not known
        self.dir_valid = os.path.join(
            self.backend.temporary_directory,
            '.auto-sklearn',
            'predictions_valid',
        )
        # test set (private test set) -- y_true not known
        self.dir_test = os.path.join(
            self.backend.temporary_directory,
            '.auto-sklearn',
            'predictions_test',
        )

        logger_name = 'EnsembleBuilder(%d):%s' % (self.seed, self.dataset_name)
        self.logger = get_logger(logger_name)

        self.start_time = 0
        self.model_fn_re = re.compile(r'_([0-9]*)_([0-9]*)\.npy')

        # already read prediction files
        # {"file name": {
        #    "ens_score": float
        #    "mtime_ens": str,
        #    "mtime_valid": str,
        #    "mtime_test": str,
        #    "seed": int,
        #    "num_run": int,
        #    Y_ENSEMBLE: np.ndarray
        #    Y_VALID: np.ndarray
        #    Y_TEST: np.ndarray
        # }
        self.read_preds = {}
        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None
        self.SAVE2DISC = True

        self.validation_performance_ = np.inf

    def run(self):
        buffer_time = 5  # TODO: Buffer time should also be used in main!?
        while True:
            time_left = self.time_limit - buffer_time
            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=int(time_left),
                mem_in_mb=self.memory_limit,
                logger=self.logger
            )(self.main)
            safe_ensemble_script()
            if safe_ensemble_script.exit_status is pynisher.MemorylimitException:
                # if ensemble script died because of memory error,
                # reduce nbest to reduce memory consumption and try it again
                if self.ensemble_nbest == 1:
                    self.logger.critical("Memory Exception -- Unable to escape from memory exception")
                else:
                    self.ensemble_nbest =  int(self.ensemble_nbest/2)
                    self.logger.warning("Memory Exception -- restart with less ensemle_nbest: %d" %(self.ensemble_nbest ))
                    # ATTENTION: main will start from scratch;
                    # all data structures are empty again
                    continue
            break

    def main(self):

        self.start_time = time.time()
        iteration = 0

        while True:

            #maximal number of iterations
            if (self.max_iterations is not None
                    and 0 < self.max_iterations <= iteration):
                self.logger.info("Terminate ensemble building because of max iterations: %d of %d",
                                 self.max_iterations,
                                 iteration)
                break

            used_time = time.time() - self.start_time
            self.logger.debug(
                'Starting iteration %d, time left: %f',
                iteration,
                self.time_limit - used_time,
            )

            # populates self.read_preds
            if not self.read_ensemble_preds():
                time.sleep(self.sleep_duration)
                continue

            selected_models = self.get_n_best_preds()
            if not selected_models:  # nothing selected
                continue

            # populates predictions in self.read_preds
            # reduces selected models if file reading failed
            n_sel_valid, n_sel_test = self.\
                get_valid_test_preds(selected_keys=selected_models)

            selected_models_set = set(selected_models)
            if selected_models_set.intersection(n_sel_test):
                selected_models = list(selected_models_set.intersection(n_sel_test))
            elif selected_models_set.intersection(n_sel_valid):
                selected_models = list(selected_models_set.intersection(n_sel_valid))
            else:
                # use selected_models only defined by ensemble data set
                pass

            # train ensemble
            ensemble = self.fit_ensemble(selected_keys=selected_models)

            if ensemble is not None:

                self.predict(set_="valid",
                             ensemble=ensemble,
                             selected_keys=n_sel_valid,
                             n_preds=len(selected_models),
                             index_run=iteration)
                # TODO if predictions fails, build the model again during the
                #  next iteration!
                self.predict(set_="test",
                             ensemble=ensemble,
                             selected_keys=n_sel_test,
                             n_preds=len(selected_models),
                             index_run=iteration)
                iteration += 1
            else:
                time.sleep(self.sleep_duration)

    def read_ensemble_preds(self):
        """
            reading predictions on ensemble building data set;
            populates self.read_preds
        """
        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

        # no validation predictions so far -- no dir
        if not os.path.isdir(self.dir_ensemble):
            self.logger.debug("No ensemble dataset prediction directory found")
            return False

        if self.shared_mode is False:
            pred_path = os.path.join(
                    self.dir_ensemble,
                    'predictions_ensemble_%s_*.npy' % self.seed)
        # pSMAC
        else:
            pred_path = os.path.join(
                    self.dir_ensemble,
                    'predictions_ensemble_*_*.npy')

        y_ens_files = glob.glob(pred_path)
        # no validation predictions so far -- no files
        if len(y_ens_files) == 0:
            self.logger.debug("Found no prediction files on ensemble data set:"
                              " %s" % pred_path)
            return False

        n_read_files = 0
        for y_ens_fn in y_ens_files:

            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy"):
                self.logger.info('Error loading file (not .npy): %s', y_ens_fn)
                continue

            match = self.model_fn_re.search(y_ens_fn)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))

            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    "ens_score": -1,
                    "mtime_ens": 0,
                    "mtime_valid": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    Y_ENSEMBLE: None,
                    Y_VALID: None,
                    Y_TEST: None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    "loaded": 0
                }

            if self.read_preds[y_ens_fn]["mtime_ens"] == os.path.getmtime(y_ens_fn):
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and score them
            try:
                with open(y_ens_fn, 'rb') as fp:
                    y_ensemble = self._read_np_fn(fp=fp)
                    score = calculate_score(solution=self.y_true_ensemble,  # y_ensemble = y_true for ensemble set
                                            prediction=y_ensemble,
                                            task_type=self.task_type,
                                            metric=self.metric,
                                            all_scoring_functions=False)

                    if self.read_preds[y_ens_fn]["ens_score"] > -1:
                        self.logger.critical(
                            'Changing ensemble score for file %s from %f to %f '
                            'because file modification time changed? %f - %f',
                            y_ens_fn,
                            self.read_preds[y_ens_fn]["ens_score"],
                            score,
                            self.read_preds[y_ens_fn]["mtime_ens"],
                            os.path.getmtime(y_ens_fn),
                        )

                    self.read_preds[y_ens_fn]["ens_score"] = score
                    self.read_preds[y_ens_fn][Y_ENSEMBLE] = y_ensemble
                    self.read_preds[y_ens_fn]["mtime_ens"] = os.path.getmtime(
                        y_ens_fn
                    )
                    self.read_preds[y_ens_fn]["loaded"] = 1

                    n_read_files += 1

            except:
                self.logger.warning(
                    'Error loading %s: %s',
                    y_ens_fn,
                    traceback.format_exc(),
                )
                self.read_preds[y_ens_fn]["ens_score"] = -1

        self.logger.debug(
            'Done reading %d new prediction files. Loaded %d predictions in '
            'total.',
            n_read_files,
            np.sum([pred["loaded"] > 0 for pred in self.read_preds.values()])
        )
        return True

    def get_n_best_preds(self):
        """
            get best n predictions (i.e., keys of self.read_preds)
            according to score on "ensemble set"
            n: self.ensemble_nbest

            Side effect: delete predictions of non-winning models
        """

        # Sort by score - higher is better!
        sorted_keys = list(reversed(sorted(
            [
                (k, v["ens_score"], v["num_run"])
                for k, v in self.read_preds.items()
            ],
            key=lambda x: x[1],
        )))
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is not run_id=0)
        dummy_score = list(filter(lambda x: x[2] == 1, sorted_keys))[0]
        self.logger.debug("Use %f as dummy score" %
                          dummy_score[1])
        sorted_keys = filter(lambda x: x[1] > dummy_score[1], sorted_keys)
        # remove Dummy Classifier
        sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
        if not sorted_keys:
            # no model left; try to use dummy score (num_run==0)
            self.logger.warning("No models better than random - "
                                "using Dummy Score!")
            sorted_keys = [
                (k, v["ens_score"], v["num_run"]) for k, v in self.read_preds.items()
                if v["seed"] == self.seed and v["num_run"] == 1
            ]
        # reload predictions if scores changed over time and a model is
        # considered to be in the top models again!
        for k, _, _ in sorted_keys[:self.ensemble_nbest]:
            if self.read_preds[k][Y_ENSEMBLE] is None:
                self.read_preds[k][Y_ENSEMBLE] = self._read_np_fn(fp=k)
                # No need to load valid and test here because they are loaded
                #  only if the model ends up in the ensemble
            self.read_preds[k]['loaded'] = 1

        # Hack ensemble_nbest to only consider models which are not
        # significantly worse than the best(and better than random)
        ensemble_n_best = None
        # TODO generalize this and re-add
        # best_loss = 1 - (sorted_keys[0][1] * 2 - 1)
        # for i in range(1, min(self.ensemble_nbest, len(sorted_keys))):
        #     current_loss = 1 - (sorted_keys[i][1] * 2 - 1)
        #     if best_loss * 1.3 < current_loss:
        #         ensemble_n_best = i
        #         self.logger.info('Reduce ensemble_nbest to %d!', ensemble_n_best)
        #         break
        if ensemble_n_best is None:
            ensemble_n_best = self.ensemble_nbest

        # reduce to keys
        sorted_keys = list(map(lambda x: x[0], sorted_keys))

        # remove loaded predictions for non-winning models
        for k in sorted_keys[ensemble_n_best:]:
            self.read_preds[k][Y_ENSEMBLE] = None
            self.read_preds[k][Y_VALID] = None
            self.read_preds[k][Y_TEST] = None
            if self.read_preds[k]['loaded'] == 1:
                self.logger.debug(
                    'Dropping model %s (%d,%d) with score %f.',
                    k,
                    self.read_preds[k]['seed'],
                    self.read_preds[k]['num_run'],
                    self.read_preds[k]['ens_score'],
                )
                self.read_preds[k]['loaded'] = 2

        # return best scored keys of self.read_preds
        return sorted_keys[:ensemble_n_best]

    def get_valid_test_preds(self, selected_keys: list):
        """
        get valid and test predictions from disc
        and store them in self.read_preds

        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_preds

        Return
        ------
        success_keys:
            all keys in selected keys for which we could read the valid and test predictions
        """
        success_keys_valid = []
        success_keys_test = []

        for k in selected_keys:
            valid_fn = glob.glob(
                os.path.join(self.dir_valid, 'predictions_valid_%d_%d.npy'
                                    % (self.read_preds[k]["seed"],
                                       self.read_preds[k]["num_run"])))
            test_fn = glob.glob(
                os.path.join(self.dir_test, 'predictions_test_%d_%d.npy' %
                                   (self.read_preds[k]["seed"],
                                    self.read_preds[k]["num_run"])))

            # TODO don't read valid and test if not changed
            if len(valid_fn) == 0:
                # self.logger.debug("Not found validation prediction file "
                #                   "(although ensemble predictions available): "
                #                   "%s" % valid_fn)
                pass
            else:
                valid_fn = valid_fn[0]
                if self.read_preds[k]["mtime_valid"] == os.path.getmtime(valid_fn) \
                        and self.read_preds[k][Y_VALID] is not None:
                    success_keys_valid.append(k)
                    continue
                try:
                    with open(valid_fn, 'rb') as fp:
                        y_valid = self._read_np_fn(fp)
                        self.read_preds[k][Y_VALID] = y_valid
                        success_keys_valid.append(k)
                        self.read_preds[k]["mtime_valid"] = os.path.getmtime(valid_fn)
                except Exception as e:
                    self.logger.warning('Error loading %s: %s',
                                        valid_fn, traceback.format_exc())

            if len(test_fn) == 0:
                # self.logger.debug("Not found test prediction file (although "
                #                   "ensemble predictions available):%s" %
                #                   test_fn)
                pass
            else:
                test_fn = test_fn[0]
                if self.read_preds[k]["mtime_test"] == \
                        os.path.getmtime(test_fn) \
                        and self.read_preds[k][Y_TEST] is not None:
                    success_keys_test.append(k)
                    continue
                try:
                    with open(test_fn, 'rb') as fp:
                        y_test = self._read_np_fn(fp)
                        self.read_preds[k][Y_TEST] = y_test
                        success_keys_test.append(k)
                        self.read_preds[k]["mtime_test"] = os.path.getmtime(test_fn)
                except Exception as e:
                    self.logger.warning('Error loading %s: %s',
                                        test_fn, traceback.format_exc())

        return success_keys_valid, success_keys_test

    def fit_ensemble(self, selected_keys:list):
        """
            fit ensemble

            Parameters
            ---------
            selected_keys: list
                list of selected keys of self.read_preds

            Returns
            -------
            ensemble: EnsembleSelection
                trained Ensemble
        """

        predictions_train = np.array([self.read_preds[k][Y_ENSEMBLE] for k in selected_keys])
        include_num_runs = [(self.read_preds[k]["seed"], self.read_preds[k]["num_run"]) for k in selected_keys]

        # check hash if ensemble training data changed
        current_hash = hash(predictions_train.data.tobytes())
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: %f",
                self.validation_performance_,
            )
            return None
        self.last_hash = current_hash

        ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            task_type=self.task_type,
            metric=self.metric,
            random_state=self.random_state,
        )

        try:
            self.logger.debug(
                "Fitting the ensemble on %d models.",
                len(predictions_train),
            )
            start_time = time.time()
            ensemble.fit(predictions_train, self.y_true_ensemble,
                         include_num_runs)
            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took %.2f seconds.",
                end_time - start_time,
            )
            self.logger.info(ensemble)
            self.validation_performance_ = min(
                self.validation_performance_,
                ensemble.get_validation_performance(),
            )

        except ValueError as e:
            self.logger.error('Caught ValueError: %s', traceback.format_exc())
            time.sleep(self.sleep_duration)
            return None
        except IndexError as e:
            self.logger.error('Caught IndexError: %s' + traceback.format_exc())
            time.sleep(self.sleep_duration)
            return None

        return ensemble

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds:int,
                index_run:int):
        """
            save preditions on ensemble, validation and test data on disc

            Parameters
            ----------
            set_: ["valid","test"]
                data split name
            ensemble: EnsembleSelection
                trained Ensemble
            selected_keys: list
                list of selected keys of self.read_preds
            n_preds: int
                number of prediction models used for ensemble building
                same number of predictions on valid and test are necessary
            index_run: int
                n-th time that ensemble predictions are written to disc

            Return
            ------
            y: np.ndarray
        """
        self.logger.debug("Predicting the %s set with the ensemble!", set_)

        # Save the ensemble for later use in the main auto-sklearn module!
        if self.SAVE2DISC:
            self.backend.save_ensemble(ensemble, index_run, self.seed)

        predictions = np.array([
            self.read_preds[k][Y_VALID if set_ == 'valid' else Y_TEST]
            for k in selected_keys
        ])

        if n_preds == predictions.shape[0]:
            y = ensemble.predict(predictions)
            if self.task_type == BINARY_CLASSIFICATION:
                y = y[:,1]
            if self.SAVE2DISC:
                self.backend.save_predictions_as_txt(
                    predictions=y,
                    subset=set_,
                    idx=index_run,
                    prefix=self.dataset_name,
                    precision=8,
                )
            return y
        else:
            self.logger.info(
                "Found inconsistent number of predictions and models (%d vs "
                "%d) for subset %s",
                predictions.shape[0],
                n_preds,
                set_,
            )
            return None
        # TODO: ADD saving of predictions on "ensemble data"

    def _read_np_fn(self, fp):
        if self.precision is "16":
            predictions = np.load(fp).astype(dtype=np.float16)
        elif self.precision is "32":
            predictions = np.load(fp).astype(dtype=np.float32)
        elif self.precision is "64":
            predictions = np.load(fp).astype(dtype=np.float64)
        else:
            predictions = np.load(fp)
        return predictions
