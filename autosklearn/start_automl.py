import os
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing

import lockfile
import numpy as np

from autosklearn.data import data_manager as data_manager
from autosklearn.metalearning import metalearning
from autosklearn.models import paramsklearn
from autosklearn.data import split_data
from autosklearn import submit_process
from autosklearn.util import stopwatch

from HPOlibConfigSpace.converters import pcs_parser

import autosklearn.util.logging_


class AutoML(multiprocessing.Process):
    def __init__(self, queue, basename, input_dir, tmp_dir, output_dir,
                 time_left_for_this_task, per_run_time_limit, log_dir=None,
                 initial_configurations_via_metalearning=25, ensemble_size=1,
                 ensemble_nbest=1, seed=1):
        super(AutoML, self).__init__()
        self.queue = queue
        self.basename = basename
        self.input_dir = input_dir
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.log_dir = log_dir
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.logger = autosklearn.util.logging_.get_logger(
            outputdir=self.log_dir,
            name="AutoML_%s_%d" % (self.basename, self.seed))

    def run(self):
        self.start_automl()

    def start_automl(self):
        # Set environment variable:
        seed = os.environ.get("AUTOSKLEARN_SEED")
        if seed is not None and int(seed) != self.seed:
            raise ValueError("It seems you have already started an instance "
                             "of AutoSklearn in this thread.")
        else:
            os.environ["AUTOSKLEARN_SEED"] = str(self.seed)

        stop = stopwatch.StopWatch()
        stop.start_task(self.basename)
        stop.start_task("LoadData")

        # == Creating a data object with data and information about it
        self.logger.debug("======== Reading and converting data ==========")
        # Encoding the labels will be done after the metafeature calculation!
        self.loaded_data_manager = data_manager.DataManager(self.basename,
                                                       self.input_dir,
                                                       verbose=True,
                                                       encode_labels=False)
        loaded_data_manager_str = str(self.loaded_data_manager).split("\n")
        for part in loaded_data_manager_str:
            self.logger.debug(part)

        # == Split dataset and store Data for the ensemble script
        X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(
            self.loaded_data_manager.data['X_train'], self.loaded_data_manager.data['Y_train'])

        true_labels_ensemble_filename = os.path.join(self.tmp_dir,
                                                     "true_labels_ensemble.npy")
        true_labels_ensemble_lock = true_labels_ensemble_filename + ".lock"
        with lockfile.LockFile(true_labels_ensemble_lock):
            if not os.path.exists(true_labels_ensemble_filename):
                np.save(true_labels_ensemble_filename, Y_ensemble)

        del X_train, X_ensemble, Y_train, Y_ensemble

        time_needed_to_load_data = stop.wall_elapsed(self.basename)
        time_left_after_reading = max(0, self.time_left_for_this_task -
                                      time_needed_to_load_data)
        self.logger.info("Remaining time after reading %s %5.2f sec" %
                    (self.basename, time_left_after_reading))

        stop.stop_task("LoadData")
        # = Create a searchspace
        stop.start_task("CreateConfigSpace")
        configspace_path = os.path.join(self.tmp_dir, "space.pcs")
        self.configuration_space = paramsklearn.get_configuration_space(
            self.loaded_data_manager.info)

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
        stop.stop_task("CreateConfigSpace")

        # == Calculate metafeatures
        stop.start_task("CalculateMetafeatures")
        categorical = [True if feat_type.lower() in ["categorical"] else False
                       for feat_type in self.loaded_data_manager.feat_type]

        if self.initial_configurations_via_metalearning <= 0:
            ml = None
        elif self.loaded_data_manager.info["task"].lower() in \
                ["multiclass.classification", "binary.classification"]:
            ml = metalearning.MetaLearning()
            self.logger.debug("Start calculating metafeatures for %s" %
                         self.loaded_data_manager.basename)
            ml.calculate_metafeatures_with_labels(self.loaded_data_manager.data["X_train"],
                                                  self.loaded_data_manager.data["Y_train"],
                                                  categorical=categorical,
                                                  dataset_name=self.loaded_data_manager.basename)
        else:
            ml = None
            self.logger.critical("Metafeatures not calculated")
        stop.stop_task("CalculateMetafeatures")
        self.logger.debug("Calculating Metafeatures (categorical attributes) took %5.2f" % stop.wall_elapsed("CalculateMetafeatures"))

        stop.start_task("OneHot")
        self.loaded_data_manager.perform1HotEncoding()
        stop.stop_task("OneHot")

        if ml is None:
            initial_configurations = []
        elif self.loaded_data_manager.info["task"].lower() in \
                ["multiclass.classification", "binary.classification"]:
            stop.start_task("CalculateMetafeaturesEncoded")
            ml.calculate_metafeatures_encoded_labels(X_train=self.loaded_data_manager.data["X_train"],
                                                     Y_train=self.loaded_data_manager.data["Y_train"],
                                                     categorical=[False] * self.loaded_data_manager.data["X_train"].shape[0],
                                                     dataset_name=self.loaded_data_manager.basename)
            stop.stop_task("CalculateMetafeaturesEncoded")
            self.logger.debug(
                "Calculating Metafeatures (encoded attributes) took %5.2fsec" %
                stop.wall_elapsed("CalculateMetafeaturesEncoded"))

            self.logger.debug(ml._metafeatures_labels.__repr__(verbosity=2))
            self.logger.debug(ml._metafeatures_encoded_labels.__repr__(verbosity=2))

            stop.start_task("InitialConfigurations")
            try:
                initial_configurations = ml.create_metalearning_string_for_smac_call(
                    self.configuration_space, self.loaded_data_manager.basename, self.loaded_data_manager.info[
                        'metric'], self.initial_configurations_via_metalearning)
            except Exception as e:
                import traceback

                self.logger.error(str(e))
                self.logger.error(traceback.format_exc())
                initial_configurations = []

            stop.stop_task("InitialConfigurations")

            self.logger.debug("Initial Configurations: (%d)", len(initial_configurations))
            for initial_configuration in initial_configurations:
                self.logger.debug(initial_configuration)
            self.logger.debug("Looking for initial configurations took %5.2fsec" %
                         stop.wall_elapsed("InitialConfigurations"))
            self.logger.info(
                "Time left for %s after finding initial configurations: %5.2fsec" %
                (self.basename, self.time_left_for_this_task -
                                stop.wall_elapsed(self.basename)))
        else:
            initial_configurations = []
            self.logger.critical("Metafeatures encoded not calculated")

        # == Pickle the data manager
        stop.start_task("StoreDatamanager")
        data_manager_path = os.path.join(self.tmp_dir,
                                         self.basename + "_Manager.pkl")
        data_manager_lockfile = data_manager_path + ".lock"
        with lockfile.LockFile(data_manager_lockfile):
            if not os.path.exists(data_manager_path):
                pickle.dump(self.loaded_data_manager,
                            open(data_manager_path, 'w'), protocol=-1)
                self.logger.debug("Pickled Datamanager at %s" %
                                  data_manager_path)
            else:
                self.logger.debug("Data manager already presend at %s" %
                                  data_manager_path)
        stop.stop_task("StoreDatamanager")

        # == RUN SMAC
        stop.start_task("runSmac")
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
        dataset = os.path.join(self.input_dir, self.basename)
        time_left_for_smac = max(0, self.time_left_for_this_task - (
                                    stop.wall_elapsed(self.basename)))
        self.logger.debug("Start SMAC with %5.2fsec time left" % time_left_for_smac)
        proc_smac, smac_call = \
            submit_process.run_smac(dataset=dataset,
                                    tmp_dir=self.tmp_dir,
                                    searchspace=configspace_path,
                                    instance_file=instance_file,
                                    limit=time_left_for_smac,
                                    cutoff_time=self.per_run_time_limit,
                                    initial_challengers=initial_configurations,
                                    seed=self.seed)
        self.logger.debug(smac_call)
        stop.stop_task("runSmac")

        # == RUN ensemble builder
        stop.start_task("runEnsemble")
        time_left_for_ensembles = max(0, self.time_left_for_this_task - (
                                         stop.wall_elapsed(self.basename)))
        self.logger.debug("Start Ensemble with %5.2fsec time left" % time_left_for_ensembles)
        proc_ensembles = \
            submit_process.run_ensemble_builder(tmp_dir=self.tmp_dir,
                                                dataset_name=self.basename,
                                                task_type=self.loaded_data_manager.info['task'],
                                                metric=self.loaded_data_manager.info['metric'],
                                                limit=time_left_for_ensembles,
                                                output_dir=self.output_dir,
                                                ensemble_size=self.ensemble_size,
                                                ensemble_nbest=self.ensemble_nbest,
                                                seed=self.seed)
        stop.stop_task("runEnsemble")

        self.queue.put([time_needed_to_load_data, data_manager_path,
                   proc_smac, proc_ensembles])
        del self.loaded_data_manager

        # Delete AutoSklearn environment variable
        del os.environ["AUTOSKLEARN_SEED"]
        return

    def configuration_space_created_hook(self):
        pass