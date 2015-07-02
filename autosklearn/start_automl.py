import os
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
import re

import lockfile
import numpy as np

from sklearn.base import BaseEstimator

from autosklearn.metalearning import metalearning
from autosklearn.models import paramsklearn, evaluator
from autosklearn.data import split_data
from autosklearn import submit_process
from autosklearn.util import stopwatch

from HPOlibConfigSpace.converters import pcs_parser

import autosklearn.util.logging_


class AutoML(multiprocessing.Process, BaseEstimator):
    def __init__(self, tmp_dir, output_dir,
                 time_left_for_this_task, per_run_time_limit, log_dir=None,
                 initial_configurations_via_metalearning=25, ensemble_size=1,
                 ensemble_nbest=1, seed=1, ml_memory_limit=3000,
                 metadata_directory=None, queue=None, keep_models=True):
        super(AutoML, self).__init__()
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.log_dir = log_dir
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.ml_memory_limit = ml_memory_limit
        self.metadata_directory = metadata_directory
        self.queue = queue
        self.keep_models = keep_models

    def run(self):
        raise NotImplementedError()

    def fit(self, X, y, task='multiclass.classification',
            metric='acc_metric', feat_type=None, dataset_name=None):
        if dataset_name is None:
            import hashlib
            m = hashlib.md5()
            m.update(X.data)
            dataset_name = m.hexdigest()
        self.basename_ = dataset_name

        self.stopwatch_ = stopwatch.StopWatch()
        self.stopwatch_.start_task(self.basename_)
        self.stopwatch_.start_task("LoadData")

        from autosklearn.data import Xy_data_manager as data_manager


        self.logger = autosklearn.util.logging_.get_logger(
            outputdir=self.log_dir,
            name="AutoML_%s_%d" % (self.basename_, self.seed))

        loaded_data_manager = data_manager.XyDataManager(
            X, y, task=task, metric=metric, feat_type=feat_type,
            encode_labels=False, dataset_name=dataset_name)

        return self._fit(loaded_data_manager)


    def fit_automl_dataset(self, basename, input_dir):
        # == Creating a data object with data and information about it
        self.basename_ = basename

        self.stopwatch_ = stopwatch.StopWatch()
        self.stopwatch_.start_task(self.basename_)

        self.logger = autosklearn.util.logging_.get_logger(
            outputdir=self.log_dir,
            name="AutoML_%s_%d" % (self.basename_, self.seed))

        self.stopwatch_.start_task("LoadData")

        from autosklearn.data import competition_data_manager as data_manager
        
        self.logger.debug("======== Reading and converting data ==========")
        # Encoding the labels will be done after the metafeature calculation!
        loaded_data_manager = data_manager.CompetitionDataManager(
            self.basename_, input_dir, verbose=True, encode_labels=False)
        loaded_data_manager_str = str(loaded_data_manager).split("\n")
        for part in loaded_data_manager_str:
            self.logger.debug(part)

        return self._fit(loaded_data_manager)

    def _fit(self, D):
        self.metric_ = D.info['metric']
        self.task_ = D.info['task']
        self.target_num_ = D.info['target_num']

        # Set environment variable:
        seed = os.environ.get("AUTOSKLEARN_SEED")
        if seed is not None and int(seed) != self.seed:
            raise ValueError("It seems you have already started an instance "
                             "of AutoSklearn in this thread.")
        else:
            os.environ["AUTOSKLEARN_SEED"] = str(self.seed)

        # == Split dataset and store Data for the ensemble script
        X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(
            D.data['X_train'], D.data['Y_train'])

        true_labels_ensemble_filename = os.path.join(self.tmp_dir,
                                                     "true_labels_ensemble.npy")
        true_labels_ensemble_lock = true_labels_ensemble_filename + ".lock"
        with lockfile.LockFile(true_labels_ensemble_lock):
            if not os.path.exists(true_labels_ensemble_filename):
                np.save(true_labels_ensemble_filename, Y_ensemble)

        del X_train, X_ensemble, Y_train, Y_ensemble

        time_needed_to_load_data = self.stopwatch_.wall_elapsed(self.basename_)
        time_left_after_reading = max(0, self.time_left_for_this_task -
                                      time_needed_to_load_data)
        self.logger.info("Remaining time after reading %s %5.2f sec" %
                    (self.basename_, time_left_after_reading))

        self.stopwatch_.stop_task("LoadData")

        # == Calculate metafeatures
        self.stopwatch_.start_task("CalculateMetafeatures")
        categorical = [True if feat_type.lower() in ["categorical"] else False
                       for feat_type in D.feat_type]

        if self.initial_configurations_via_metalearning <= 0:
            ml = None
        elif D.info["task"].lower() in \
                ["multiclass.classification", "binary.classification"]:
            ml = metalearning.MetaLearning()
            self.logger.debug("Start calculating metafeatures for %s" %
                              self.basename_)
            ml.calculate_metafeatures_with_labels(D.data["X_train"],
                                                  D.data["Y_train"],
                                                  categorical=categorical,
                                                  dataset_name=self.basename_)
        else:
            ml = None
            self.logger.critical("Metafeatures not calculated")
        self.stopwatch_.stop_task("CalculateMetafeatures")
        self.logger.debug("Calculating Metafeatures (categorical attributes) took %5.2f" % self.stopwatch_.wall_elapsed("CalculateMetafeatures"))

        self.stopwatch_.start_task("OneHot")
        D.perform1HotEncoding()
        self.ohe_ = D.encoder_
        self.stopwatch_.stop_task("OneHot")

        # == Pickle the data manager
        self.stopwatch_.start_task("StoreDatamanager")
        data_manager_path = os.path.join(self.tmp_dir,
                                         self.basename_ + "_Manager.pkl")
        data_manager_lockfile = data_manager_path + ".lock"
        with lockfile.LockFile(data_manager_lockfile):
            if not os.path.exists(data_manager_path):
                pickle.dump(D,
                            open(data_manager_path, 'w'), protocol=-1)
                self.logger.debug("Pickled Datamanager at %s" %
                                  data_manager_path)
            else:
                self.logger.debug("Data manager already presend at %s" %
                                  data_manager_path)
        self.stopwatch_.stop_task("StoreDatamanager")

        # = Create a searchspace
        self.stopwatch_.start_task("CreateConfigSpace")
        configspace_path = os.path.join(self.tmp_dir, "space.pcs")
        self.configuration_space = paramsklearn.get_configuration_space(
            D.info)

        self.configuration_space_created_hook()

        sp_string = pcs_parser.write(self.configuration_space)
        configuration_space_lockfile = configspace_path + ".lock"
        with lockfile.LockFile(configuration_space_lockfile):
            if not os.path.exists(configspace_path):
                with open(configspace_path, "w") as fh:
                    fh.write(sp_string)
                self.logger.debug("Configuration space written to %s" %
                                  configspace_path)
            else:
                self.logger.debug("Configuration space already present at %s" %
                                  configspace_path)
        self.stopwatch_.stop_task("CreateConfigSpace")

        if ml is None:
            initial_configurations = []
        elif D.info["task"].lower() in \
                ["multiclass.classification", "binary.classification"]:
            self.stopwatch_.start_task("CalculateMetafeaturesEncoded")
            ml.calculate_metafeatures_encoded_labels(X_train=D.data["X_train"],
                                                     Y_train=D.data["Y_train"],
                                                     categorical=[False] * D.data["X_train"].shape[0],
                                                     dataset_name=self.basename_)
            self.stopwatch_.stop_task("CalculateMetafeaturesEncoded")
            self.logger.debug(
                "Calculating Metafeatures (encoded attributes) took %5.2fsec" %
                self.stopwatch_.wall_elapsed("CalculateMetafeaturesEncoded"))

            self.logger.debug(ml._metafeatures_labels.__repr__(verbosity=2))
            self.logger.debug(ml._metafeatures_encoded_labels.__repr__(verbosity=2))

            self.stopwatch_.start_task("InitialConfigurations")
            try:
                initial_configurations = ml.create_metalearning_string_for_smac_call(
                    self.configuration_space, self.basename_, self.metric_,
                    self.task_, True if D.info['is_sparse'] == 1 else False,
                    self.initial_configurations_via_metalearning, self.metadata_directory)
            except Exception as e:
                import traceback

                self.logger.error(str(e))
                self.logger.error(traceback.format_exc())
                initial_configurations = []

            self.stopwatch_.stop_task("InitialConfigurations")

            self.logger.debug("Initial Configurations: (%d)", len(initial_configurations))
            for initial_configuration in initial_configurations:
                self.logger.debug(initial_configuration)
            self.logger.debug("Looking for initial configurations took %5.2fsec" %
                              self.stopwatch_.wall_elapsed("InitialConfigurations"))
            self.logger.info(
                "Time left for %s after finding initial configurations: %5.2fsec" %
                (self.basename_, self.time_left_for_this_task -
                 self.stopwatch_.wall_elapsed(self.basename_)))
        else:
            initial_configurations = []
            self.logger.critical("Metafeatures encoded not calculated")

        # == Set up a directory where all the trained models will be pickled to
        if self.keep_models:
            self.model_directory_ = os.path.join(self.tmp_dir,
                                                 "models_%d" % self.seed)
            os.mkdir(self.model_directory_)
        self.ensemble_indices_directory_ = os.path.join(self.tmp_dir,
                                                        "ensemble_indices_%d" % self.seed)
        os.mkdir(self.ensemble_indices_directory_)

        # == RUN SMAC
        self.stopwatch_.start_task("runSmac")
        # = Create an empty instance file
        instance_file = os.path.join(self.tmp_dir, "instances.txt")
        instance_file_lock = instance_file + ".lock"
        with lockfile.LockFile(instance_file_lock):
            if not os.path.exists(instance_file_lock):
                with open(instance_file, "w") as fh:
                    fh.write("holdout")
                self.logger.debug("Created instance file %s" % instance_file)
            else:
                self.logger.debug("Instance file already present at %s" % instance_file)

        # = Start SMAC
        time_left_for_smac = max(0, self.time_left_for_this_task - (
            self.stopwatch_.wall_elapsed(self.basename_)))
        self.logger.debug("Start SMAC with %5.2fsec time left" % time_left_for_smac)
        proc_smac, smac_call = \
            submit_process.run_smac(dataset_name=self.basename_,
                                    dataset=data_manager_path,
                                    tmp_dir=self.tmp_dir,
                                    searchspace=configspace_path,
                                    instance_file=instance_file,
                                    limit=time_left_for_smac,
                                    cutoff_time=self.per_run_time_limit,
                                    initial_challengers=initial_configurations,
                                    memory_limit=self.ml_memory_limit,
                                    seed=self.seed)
        self.logger.debug(smac_call)
        self.stopwatch_.stop_task("runSmac")

        # == RUN ensemble builder
        self.stopwatch_.start_task("runEnsemble")
        time_left_for_ensembles = max(0, self.time_left_for_this_task - (
            self.stopwatch_.wall_elapsed(self.basename_)))
        self.logger.debug("Start Ensemble with %5.2fsec time left" % time_left_for_ensembles)
        proc_ensembles = \
            submit_process.run_ensemble_builder(tmp_dir=self.tmp_dir,
                                                dataset_name=self.basename_,
                                                task_type=self.task_,
                                                metric=self.metric_,
                                                limit=time_left_for_ensembles,
                                                output_dir=self.output_dir,
                                                ensemble_size=self.ensemble_size,
                                                ensemble_nbest=self.ensemble_nbest,
                                                seed=self.seed,
                                                ensemble_indices_output_dir=self.ensemble_indices_directory_)
        self.stopwatch_.stop_task("runEnsemble")

        del D

        if self.queue is not None:
            self.queue.put([time_needed_to_load_data, data_manager_path,
                            proc_smac, proc_ensembles])
        else:
            proc_smac.wait()
            proc_ensembles.wait()

        # Delete AutoSklearn environment variable
        del os.environ["AUTOSKLEARN_SEED"]
        return self

    def predict(self, X):
        if self.keep_models is not True:
            raise ValueError("Predict can only be called if 'keep_models==True'")

        model_files = os.listdir(self.model_directory_)
        models = []
        for model_file in model_files:
            model_file = os.path.join(self.model_directory_, model_file)
            with open(model_file) as fh:
                models.append(pickle.load(fh))

        if len(models) == 0:
            raise ValueError("No models fitted!")

        if self.ohe_ is not None:
            X = self.ohe_._transform(X)

        indices_files = sorted(os.listdir(self.ensemble_indices_directory_))
        indices_file = os.path.join(self.ensemble_indices_directory_,
                                    indices_files[-1])
        with open(indices_file) as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        predictions = []
        for model, model_file in zip(models, model_files):
            num_run = int(model_file.split(".")[0])

            if num_run not in ensemble_members_run_numbers:
                continue

            weight = ensemble_members_run_numbers[num_run]

            X_ = X.copy()
            if "regression" in self.task_:
                prediction = model.predict(X_)
            else:
                prediction = model.predict_proba(X_)
            predictions.append(prediction * weight)

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def score(self, X, y):
        prediction = self.predict(X)
        return evaluator.calculate_score(y, prediction, self.task_,
                                         self.metric_, self.target_num_)

    def configuration_space_created_hook(self):
        pass

    def get_params(self, deep=True):
        raise NotImplementedError("auto-sklearn does not implement "
                                  "get_params() because it is not intended to "
                                  "be optimized.")

    def set_params(self, deep=True):
        raise NotImplementedError("auto-sklearn does not implement "
                                  "set_params() because it is not intended to "
                                  "be optimized.")