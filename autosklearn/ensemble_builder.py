# -*- encoding: utf-8 -*-
import math
import numbers
import multiprocessing
import glob
import gzip
import os
import re
import time
import traceback
from typing import Optional, Union

import numpy as np
import pandas as pd
import pynisher
import lockfile
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
            ensemble_size: int = 10,
            ensemble_nbest: int = 100,
            max_models_on_disc: int = 100,
            performance_range_threshold: float = 0,
            seed: int = 1,
            shared_mode: bool = False,
            max_iterations: int = None,
            precision: int = 32,
            sleep_duration: int = 2,
            memory_limit: Optional[int] = 1024,
            read_at_most: int = 5,
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            queue: multiprocessing.Queue = None
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
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: int
               Defines the maximum number of models that are kept in the disc.
               If int, it must be greater or equal than 1, and dictates the max number of
               models to keep.
               If float, it will be interpreted as the max megabytes allowed of disc space. That
               is, if the number of ensemble candidates require more disc space than this float
               value, the worst models will be deleted to keep within this budget.
               Models and predictions of the worst-performing models will be deleted then.
               If None, the feature is disabled.
               It defines an upper bound on the models that can be used in the ensemble.
            performance_range_threshold: float
                Keep only models that are better than:
                    dummy + (best - dummy)*performance_range_threshold
                E.g dummy=2, best=4, thresh=0.5 --> only consider models with score > 3
                Will at most return the minimum between ensemble_nbest models,
                and max_models_on_disc. Might return less
            seed: int
                random seed
                if set to -1, read files with any seed (e.g., for shared model mode)
            shared_model: bool
                auto-sklearn used shared model mode (aka pSMAC)
            max_iterations: int
                maximal number of iterations to run this script
                (default None --> deactivated)
            precision: [16,32,64,128]
                precision of floats to read the predictions
            sleep_duration: int
                duration of sleeping time between two iterations of this script (in sec)
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
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
        self.performance_range_threshold = performance_range_threshold

        if isinstance(ensemble_nbest, numbers.Integral) and ensemble_nbest < 1:
            raise ValueError("Integer ensemble_nbest has to be larger 1: %s" %
                             ensemble_nbest)
        elif not isinstance(ensemble_nbest, numbers.Integral):
            if ensemble_nbest < 0 or ensemble_nbest > 1:
                raise ValueError(
                    "Float ensemble_nbest best has to be >= 0 and <= 1: %s" %
                    ensemble_nbest)

        self.ensemble_nbest = ensemble_nbest

        # max_models_on_disc can be a float, in such case we need to
        # remember the user specified Megabytes and translate this to
        # max number of ensemble models. max_resident_models keeps the
        # maximum number of models in disc
        if max_models_on_disc is not None and max_models_on_disc < 0:
            raise ValueError(
                "max_models_on_disc has to be a positive number or None"
            )
        self.max_models_on_disc = max_models_on_disc
        self.max_resident_models = None

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
        self.dir_models = os.path.join(
            self.backend.temporary_directory,
            '.auto-sklearn',
            'models',
        )
        logger_name = 'EnsembleBuilder(%d):%s' % (self.seed, self.dataset_name)
        self.logger = get_logger(logger_name)
        if ensemble_nbest == 1:
            self.logger.debug("Behaviour depends on int/float: %s, %s (ensemble_nbest, type)" %
                              (ensemble_nbest, type(ensemble_nbest)))

        self.start_time = 0
        self.model_fn_re = re.compile(r'_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)\.npy')

        # already read prediction files
        # {"file name": {
        #    "ens_score": float
        #    "mtime_ens": str,
        #    "mtime_valid": str,
        #    "mtime_test": str,
        #    "seed": int,
        #    "num_run": int,
        #    "deleted": bool,
        #    Y_ENSEMBLE: np.ndarray
        #    Y_VALID: np.ndarray
        #    Y_TEST: np.ndarray
        #    }
        # }
        self.read_preds = {}
        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None
        self.SAVE2DISC = True

        # hidden feature which can be activated via an environment variable. This keeps all
        # models and predictions which have ever been a candidate. This is necessary to post-hoc
        # compute the whole ensemble building trajectory.
        self._has_been_candidate = set()

        self.validation_performance_ = np.inf

        # Track the ensemble performance
        self.datamanager = self.backend.load_datamanager()
        self.y_valid = self.datamanager.data.get('Y_valid')
        self.y_test = self.datamanager.data.get('Y_test')

        # Support for tracking the performance across time
        # A Queue is needed to handle multiprocessing, not only
        # internally for pynisher calls, but to return data
        # to the main process
        # Hence, because we are using three different processes,
        # the below strategy prevents MemoryErrors. That is,
        # without clearly isolating the queue with a manger,
        # we run into a threading MemoryError
        if queue is None:
            mgr = multiprocessing.Manager()
            mgr.Namespace()
            self.queue = mgr.Queue()
        else:
            self.queue = queue
        self.queue.put([])
        self.queue.get()

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
                if isinstance(self.ensemble_nbest, numbers.Integral) and \
                        self.ensemble_nbest == 1:
                    self.logger.critical(
                        "Memory Exception -- Unable to further reduce the number of ensemble "
                        "members -- please restart Auto-sklearn with a higher value for the "
                        "argument 'ensemble_memory_limit' (current limit is {} MB)."
                        "".format(self.memory_limit)
                    )
                else:
                    if isinstance(self.ensemble_nbest, numbers.Integral):
                        self.ensemble_nbest = int(self.ensemble_nbest / 2)
                    else:
                        self.ensemble_nbest = self.ensemble_nbest / 2
                    self.logger.warning("Memory Exception -- restart with "
                                        "less ensemble_nbest: %d" % self.ensemble_nbest)
                    # ATTENTION: main will start from scratch;
                    # all data structures are empty again
                    continue
            break

    def main(self, return_pred=False):
        """

        :param return_pred:
            return tuple with last valid, test predictions
        :return:
        """
        self.start_time = time.time()
        iteration = 0
        valid_pred, test_pred = None, None
        while True:

            # maximal number of iterations
            if (self.max_iterations is not None
                    and 0 < self.max_iterations <= iteration):
                self.logger.info("Terminate ensemble building because of max iterations: %d of %d",
                                 self.max_iterations,
                                 iteration)
                break
            iteration += 1

            used_time = time.time() - self.start_time
            self.logger.debug(
                'Starting iteration %d, time left: %f',
                iteration,
                self.time_limit - used_time,
            )

            # populates self.read_preds
            if not self.score_ensemble_preds():
                time.sleep(self.sleep_duration)
                continue

            # Only the models with the n_best predictions are candidates
            # to be in the ensemble
            candidate_models = self.get_n_best_preds()
            if not candidate_models:  # no candidates yet
                continue

            # populates predictions in self.read_preds
            # reduces selected models if file reading failed
            n_sel_valid, n_sel_test = self. \
                get_valid_test_preds(selected_keys=candidate_models)

            # If valid/test predictions loaded, then reduce candidate models to this set
            if len(n_sel_test) != 0 and len(n_sel_valid) != 0 \
                    and len(set(n_sel_valid).intersection(set(n_sel_test))) == 0:
                # Both n_sel_* have entries, but there is no overlap, this is critical
                self.logger.error("n_sel_valid and n_sel_test are not empty, but do "
                                  "not overlap")
                time.sleep(self.sleep_duration)
                continue

            # If any of n_sel_* is not empty and overlaps with candidate_models,
            # then ensure candidate_models AND n_sel_test are sorted the same
            candidate_models_set = set(candidate_models)
            if candidate_models_set.intersection(n_sel_valid).intersection(n_sel_test):
                candidate_models = sorted(list(candidate_models_set.intersection(
                    n_sel_valid).intersection(n_sel_test)))
                n_sel_test = candidate_models
                n_sel_valid = candidate_models
            elif candidate_models_set.intersection(n_sel_valid):
                candidate_models = sorted(list(candidate_models_set.intersection(
                    n_sel_valid)))
                n_sel_valid = candidate_models
            elif candidate_models_set.intersection(n_sel_test):
                candidate_models = sorted(list(candidate_models_set.intersection(
                    n_sel_test)))
                n_sel_test = candidate_models
            else:
                # This has to be the case
                n_sel_test = []
                n_sel_valid = []

            if os.environ.get('ENSEMBLE_KEEP_ALL_CANDIDATES'):
                for candidate in candidate_models:
                    self._has_been_candidate.add(candidate)

            # train ensemble
            ensemble = self.fit_ensemble(selected_keys=candidate_models)
            if ensemble is not None:
                train_pred = self.predict(set_="train",
                                          ensemble=ensemble,
                                          selected_keys=candidate_models,
                                          n_preds=len(candidate_models),
                                          index_run=iteration)
                # We can't use candidate_models here, as n_sel_* might be empty
                valid_pred = self.predict(set_="valid",
                                          ensemble=ensemble,
                                          selected_keys=n_sel_valid,
                                          n_preds=len(candidate_models),
                                          index_run=iteration)
                # TODO if predictions fails, build the model again during the
                #  next iteration!
                test_pred = self.predict(set_="test",
                                         ensemble=ensemble,
                                         selected_keys=n_sel_test,
                                         n_preds=len(candidate_models),
                                         index_run=iteration)

                # Add a score to run history to see ensemble progress
                self._add_ensemble_trajectory(
                    train_pred,
                    valid_pred,
                    test_pred
                )
            else:
                time.sleep(self.sleep_duration)

        if return_pred:
            return valid_pred, test_pred

    def get_disk_consumption(self, pred_path):
        """
        gets the cost of a model being on disc
        """

        match = self.model_fn_re.search(pred_path)
        if not match:
            raise ValueError("Invalid path format %s" % pred_path)
        _full_name = match.group(0)
        _seed = match.group(1)
        _num_run = match.group(2)
        _budget = match.group(3)

        # Besides the prediction, we have to take care of three other files: model,
        # validation and test.
        model_name = '%s.%s.%s.model' % (_seed, _num_run, _budget)
        model_path = os.path.join(self.dir_models, model_name)
        pred_valid_name = 'predictions_valid' + _full_name
        pred_valid_path = os.path.join(self.dir_valid, pred_valid_name)
        pred_test_name = 'predictions_test' + _full_name
        pred_test_path = os.path.join(self.dir_test, pred_test_name)

        paths = [model_path, pred_path]
        if os.path.exists(pred_valid_path):
            paths.append(pred_valid_path)
        if os.path.exists(pred_test_path):
            paths.append(pred_test_path)
        this_model_cost = sum([os.path.getsize(path) for path in paths])

        # get the megabytes
        return round(this_model_cost / math.pow(1024, 2), 2)

    def score_ensemble_preds(self):
        """
            score predictions on ensemble building data set;
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
                glob.escape(self.dir_ensemble),
                'predictions_ensemble_%s_*_*.npy*' % self.seed,
            )
        # pSMAC
        else:
            pred_path = os.path.join(
                glob.escape(self.dir_ensemble),
                'predictions_ensemble_*_*_*.npy*',
            )

        y_ens_files = glob.glob(pred_path)
        y_ens_files = [y_ens_file for y_ens_file in y_ens_files
                       if y_ens_file.endswith('.npy') or y_ens_file.endswith('.npy.gz')]
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug("Found no prediction files on ensemble data set:"
                              " %s" % pred_path)
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

        n_read_files = 0
        # Now read file wrt to num_run
        for y_ens_fn, match, _seed, _num_run, _budget in \
                sorted(to_read, key=lambda x: x[3]):
            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info('Error loading file (not .npy or .npy.gz): %s', y_ens_fn)
                continue

            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    "ens_score": -1,
                    "mtime_ens": 0,
                    "mtime_valid": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
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
                y_ensemble = self._read_np_fn(y_ens_fn)
                score = calculate_score(solution=self.y_true_ensemble,
                                        prediction=y_ensemble,
                                        task_type=self.task_type,
                                        metric=self.metric,
                                        all_scoring_functions=False)

                if self.read_preds[y_ens_fn]["ens_score"] > -1:
                    self.logger.debug(
                        'Changing ensemble score for file %s from %f to %f '
                        'because file modification time changed? %f - %f',
                        y_ens_fn,
                        self.read_preds[y_ens_fn]["ens_score"],
                        score,
                        self.read_preds[y_ens_fn]["mtime_ens"],
                        os.path.getmtime(y_ens_fn),
                    )

                self.read_preds[y_ens_fn]["ens_score"] = score

                # It is not needed to create the object here
                # To save memory, we just score the object.
                # self.read_preds[y_ens_fn][Y_ENSEMBLE] = y_ensemble
                self.read_preds[y_ens_fn]["mtime_ens"] = os.path.getmtime(
                    y_ens_fn
                )
                self.read_preds[y_ens_fn]["loaded"] = 2
                self.read_preds[y_ens_fn]["disc_space_cost_mb"] = self.get_disk_consumption(
                    y_ens_fn
                )

                n_read_files += 1

            except Exception:
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

            Side effects:
                ->Define the n-best models to use in ensemble
                ->Only the best models are loaded
                ->Any model that is not best is candidate to deletion
                  if max models in disc is exceeded.
        """

        sorted_keys = self._get_list_of_sorted_preds()

        # number of models available
        num_keys = len(sorted_keys)
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is no run_id=0)
        dummy_scores = list(filter(lambda x: x[2] == 1, sorted_keys))
        # number of dummy models
        num_dummy = len(dummy_scores)
        dummy_score = dummy_scores[0]
        self.logger.debug("Use %f as dummy score" % dummy_score[1])
        sorted_keys = filter(lambda x: x[1] > dummy_score[1], sorted_keys)
        # remove Dummy Classifier
        sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
        if not sorted_keys:
            # no model left; try to use dummy score (num_run==0)
            # log warning when there are other models but not better than dummy model
            if num_keys > num_dummy:
                self.logger.warning("No models better than random - using Dummy Score!"
                                    "Number of models besides current dummy model: %d. "
                                    "Number of dummy models: %d",
                                    num_keys - 1,
                                    num_dummy)
            sorted_keys = [
                (k, v["ens_score"], v["num_run"]) for k, v in self.read_preds.items()
                if v["seed"] == self.seed and v["num_run"] == 1
            ]
        # reload predictions if scores changed over time and a model is
        # considered to be in the top models again!
        if not isinstance(self.ensemble_nbest, numbers.Integral):
            # Transform to number of models to keep. Keep at least one
            keep_nbest = max(1, min(len(sorted_keys),
                                    int(len(sorted_keys) * self.ensemble_nbest)))
            self.logger.debug(
                "Library pruning: using only top %f percent of the models for ensemble "
                "(%d out of %d)",
                self.ensemble_nbest * 100, keep_nbest, len(sorted_keys)
            )
        else:
            # Keep only at most ensemble_nbest
            keep_nbest = min(self.ensemble_nbest, len(sorted_keys))
            self.logger.debug("Library Pruning: using for ensemble only "
                              " %d (out of %d) models" % (keep_nbest, len(sorted_keys)))

        # If max_models_on_disc is None, do nothing
        # One can only read at most max_models_on_disc models
        if self.max_models_on_disc is not None:
            if not isinstance(self.max_models_on_disc, numbers.Integral):
                consumption = [
                    [
                        v["ens_score"],
                        v["disc_space_cost_mb"],
                    ] for v in self.read_preds.values() if v["disc_space_cost_mb"] is not None
                ]
                max_consumption = max(i[1] for i in consumption)

                # We are pessimistic with the consumption limit indicated by
                # max_models_on_disc by 1 model. Such model is assumed to spend
                # max_consumption megabytes
                if (sum(i[1] for i in consumption) + max_consumption) > self.max_models_on_disc:

                    # just leave the best -- higher is better!
                    # This list is in descending order, to preserve the best models
                    sorted_cum_consumption = np.cumsum([
                        i[1] for i in list(reversed(sorted(consumption)))
                    ])
                    max_models = np.argmax(sorted_cum_consumption > self.max_models_on_disc)

                    # Make sure that at least 1 model survives
                    self.max_resident_models = max(1, max_models)
                    self.logger.warning(
                        "Limiting num of models via float max_models_on_disc={}"
                        " as accumulated={} worst={} num_models={}".format(
                            self.max_models_on_disc,
                            (sum(i[1] for i in consumption) + max_consumption),
                            max_consumption,
                            self.max_resident_models
                        )
                    )
            else:
                self.max_resident_models = self.max_models_on_disc

        if self.max_resident_models is not None and keep_nbest > self.max_resident_models:
            self.logger.debug(
                "Restricting the number of models to %d instead of %d due to argument "
                "max_models_on_disc",
                self.max_resident_models, keep_nbest,
            )
            keep_nbest = self.max_resident_models

        # consider performance_range_threshold
        if self.performance_range_threshold > 0:
            best_score = sorted_keys[0][1]
            min_score = dummy_score[1]
            min_score += (best_score - min_score) * self.performance_range_threshold
            if sorted_keys[keep_nbest - 1][1] < min_score:
                # We can further reduce number of models
                # since worst model is worse than thresh
                for i in range(0, keep_nbest):
                    # Look at most at keep_nbest models,
                    # but always keep at least one model
                    current_score = sorted_keys[i][1]
                    if current_score <= min_score:
                        self.logger.debug("Dynamic Performance range: "
                                          "Further reduce from %d to %d models",
                                          keep_nbest, max(1, i))
                        keep_nbest = max(1, i)
                        break
        ensemble_n_best = keep_nbest

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

        # Load the predictions for the winning
        for k in sorted_keys[:ensemble_n_best]:
            if self.read_preds[k][Y_ENSEMBLE] is None:
                self.read_preds[k][Y_ENSEMBLE] = self._read_np_fn(k)
                # No need to load valid and test here because they are loaded
                #  only if the model ends up in the ensemble
            self.read_preds[k]['loaded'] = 1

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
            all keys in selected keys for which we could read the valid and
            test predictions
        """
        success_keys_valid = []
        success_keys_test = []

        for k in selected_keys:
            valid_fn = glob.glob(
                os.path.join(
                    glob.escape(self.dir_valid),
                    'predictions_valid_%d_%d_%s.npy*' % (
                        self.read_preds[k]["seed"],
                        self.read_preds[k]["num_run"],
                        self.read_preds[k]["budget"],
                    )
                )
            )
            valid_fn = [vfn for vfn in valid_fn if vfn.endswith('.npy') or vfn.endswith('.npy.gz')]
            test_fn = glob.glob(
                os.path.join(
                    glob.escape(self.dir_test),
                    'predictions_test_%d_%d_%s.npy*' % (
                        self.read_preds[k]["seed"],
                        self.read_preds[k]["num_run"],
                        self.read_preds[k]["budget"]
                    )
                )
            )
            test_fn = [tfn for tfn in test_fn if tfn.endswith('.npy') or tfn.endswith('.npy.gz')]

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
                    y_valid = self._read_np_fn(valid_fn)
                    self.read_preds[k][Y_VALID] = y_valid
                    success_keys_valid.append(k)
                    self.read_preds[k]["mtime_valid"] = os.path.getmtime(valid_fn)
                except Exception:
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
                    y_test = self._read_np_fn(test_fn)
                    self.read_preds[k][Y_TEST] = y_test
                    success_keys_test.append(k)
                    self.read_preds[k]["mtime_test"] = os.path.getmtime(test_fn)
                except Exception:
                    self.logger.warning('Error loading %s: %s',
                                        test_fn, traceback.format_exc())

        return success_keys_valid, success_keys_test

    def fit_ensemble(self, selected_keys: list):
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
        include_num_runs = [
            (
                self.read_preds[k]["seed"],
                self.read_preds[k]["num_run"],
                self.read_preds[k]["budget"],
            )
            for k in selected_keys]

        # check hash if ensemble training data changed
        current_hash = hash(predictions_train.data.tobytes())
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: %f",
                self.validation_performance_,
            )

            # Delete files of non-candidate models
            if self.max_resident_models is not None:
                self._delete_excess_models()

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

        except ValueError:
            self.logger.error('Caught ValueError: %s', traceback.format_exc())
            time.sleep(self.sleep_duration)
            return None
        except IndexError:
            self.logger.error('Caught IndexError: %s' + traceback.format_exc())
            time.sleep(self.sleep_duration)
            return None

        # Delete files of non-candidate models
        if self.max_resident_models is not None:
            self._delete_excess_models()

        return ensemble

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds: int,
                index_run: int):
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

        if set_ == 'valid':
            pred_set = Y_VALID
        elif set_ == 'test':
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE
        predictions = np.array([
            self.read_preds[k][pred_set]
            for k in selected_keys
        ])

        if n_preds == predictions.shape[0]:
            y = ensemble.predict(predictions)
            if self.task_type == BINARY_CLASSIFICATION:
                y = y[:, 1]
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

    def get_ensemble_history(self):
        """
        Getter method to obtain the performance of the ensemble
        building process across time

        Return
        ----------
        dict that tracks the performance of the ensemble
        building process on testing/training sets

        """
        ensemble_history = []
        while(self.queue.qsize()):
            ensemble_history.append(self.queue.get())
        return ensemble_history

    def _add_ensemble_trajectory(self, train_pred, valid_pred, test_pred):
        """
        Records a snapshot of how the performance look at a given training
        time.

        Parameters
        ----------
        ensemble: EnsembleSelection
            The ensemble selection object to record
        valid_pred: np.ndarray
            The predictions on the validation set using ensemble
        test_pred: np.ndarray
            The predictions on the test set using ensemble

        """
        performance_stamp = {
            'Timestamp': pd.Timestamp.now(),
            'ensemble_optimization_score': calculate_score(
                solution=self.y_true_ensemble,
                prediction=train_pred,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=False
            )
        }
        if valid_pred is not None:
            # TODO: valid_pred are a legacy from competition manager
            # and this if never happens. Re-evaluate Y_valid support
            performance_stamp['ensemble_val_score'] = calculate_score(
                solution=self.y_valid,
                prediction=valid_pred,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=False
            )

        # In case test_pred was provided
        if test_pred is not None:
            performance_stamp['ensemble_test_score'] = calculate_score(
                solution=self.y_test,
                prediction=test_pred,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=False
            )

        self.queue.put(performance_stamp)

    def _get_list_of_sorted_preds(self):
        """
            Returns a list of sorted predictions in descending order
            Predictions are taken from self.read_preds.

            Parameters
            ----------
            None

            Return
            ------
            sorted_keys: list
        """
        # Sort by score - higher is better!
        # First sort by num_run
        sorted_keys = list(reversed(sorted(
            [
                (k, v["ens_score"], v["num_run"])
                for k, v in self.read_preds.items()
            ],
            key=lambda x: x[2],
        )))
        # Then by score
        sorted_keys = list(reversed(sorted(sorted_keys, key=lambda x: x[1])))
        return sorted_keys

    def _delete_excess_models(self):
        """
            Deletes models excess models on disc. self.max_models_on_disc
            defines the upper limit on how many models to keep.
            Any additional model with a worst score than the top
            self.max_models_on_disc is deleted.

        """

        # Obtain a list of sorted pred keys
        sorted_keys = self._get_list_of_sorted_preds()
        sorted_keys = list(map(lambda x: x[0], sorted_keys))

        if len(sorted_keys) <= self.max_resident_models:
            # Don't waste time if not enough models to delete
            return

        # The top self.max_resident_models models would be the candidates
        # Any other low performance model will be deleted
        # The list is in ascending order of score
        candidates = sorted_keys[:self.max_resident_models]

        # Loop through the files currently in the directory
        for pred_path in self.y_ens_files:

            # Do not delete candidates
            if pred_path in candidates:
                continue

            if pred_path in self._has_been_candidate:
                continue

            match = self.model_fn_re.search(pred_path)
            _full_name = match.group(0)
            _seed = match.group(1)
            _num_run = match.group(2)
            _budget = match.group(3)

            # Do not delete the dummy prediction
            if int(_num_run) == 1:
                continue

            # Besides the prediction, we have to take care of three other files: model,
            # validation and test.
            model_name = '%s.%s.%s.model' % (_seed, _num_run, _budget)
            model_path = os.path.join(self.dir_models, model_name)
            pred_valid_name = 'predictions_valid' + _full_name
            pred_valid_path = os.path.join(self.dir_valid, pred_valid_name)
            pred_test_name = 'predictions_test' + _full_name
            pred_test_path = os.path.join(self.dir_test, pred_test_name)

            paths = [model_path, pred_path]
            if os.path.exists(pred_valid_path):
                paths.append(pred_valid_path)
            if os.path.exists(pred_test_path):
                paths.append(pred_test_path)

            # Lets lock all the files "at once" to avoid weird race conditions. Also,
            # we either delete all files of a model (model, prediction, validation
            # and test), or delete none. This makes it easier to keep track of which
            # models have indeed been correctly removed.
            locks = [lockfile.LockFile(path) for path in paths]
            try:
                for lock in locks:
                    lock.acquire()
            except Exception as e:
                if isinstance(e, lockfile.AlreadyLocked):
                    # If the file is already locked, we deal with it later. Not a big deal
                    self.logger.info(
                        'Model %s is already locked. Skipping it for now.', model_name)
                else:
                    # Other exceptions, however, should not occur.
                    # The message bellow is asserted in test_delete_excess_models()
                    self.logger.error(
                        'Failed to lock model %s files due to error %s', model_name, e)
                for lock in locks:
                    if lock.i_am_locking():
                        lock.release()
                continue

            # Delete files if model is not a candidate AND prediction is old. We check if
            # the prediction is old to avoid deleting a model that hasn't been appreciated
            # by self.get_n_best_preds() yet.
            original_timestamp = self.read_preds[pred_path]['mtime_ens']
            current_timestamp = os.path.getmtime(pred_path)
            if current_timestamp == original_timestamp:
                # The messages logged here are asserted in
                # test_delete_excess_models(). Edit with care.
                try:
                    for path in paths:
                        os.remove(path)
                    self.logger.info(
                        "Deleted files of non-candidate model %s", model_name)
                except Exception as e:
                    self.logger.error(
                        "Failed to delete files of non-candidate model %s due"
                        " to error %s", model_name, e)

            # If we reached this point, all locks were done by this thread. So no need
            # to check "lock.i_am_locking()" here.
            for lock in locks:
                lock.release()

    def _read_np_fn(self, path):

        # Support for string precision
        if isinstance(self.precision, str):
            precision = int(self.precision)
            self.logger.warning("Interpreted str-precision as {}".format(
                precision
            ))
        else:
            precision = self.precision

        if path.endswith("gz"):
            open_method = gzip.open
        elif path.endswith("npy"):
            open_method = open
        else:
            raise ValueError("Unknown filetype %s" % path)
        with open_method(path, 'rb') as fp:
            if precision == 16:
                predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float16)
            elif precision == 32:
                predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float32)
            elif precision == 64:
                predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float64)
            else:
                predictions = np.load(fp, allow_pickle=True)
            return predictions
