import os
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from autosklearn.data import data_manager as data_manager
from autosklearn.metalearning import metalearning
from autosklearn.models import paramsklearn
from autosklearn.data import split_data
from autosklearn import submit_process
from autosklearn.util import stopwatch

from HPOlibConfigSpace.converters import pcs_parser
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

import autosklearn.util.logging_


def start_automl_on_dataset(basename, input_dir, tmp_dataset_dir, output_dir,
                            time_left_for_this_task, queue, log_dir=None,
                            initial_configurations_via_metalearning=25,
                            ensemble_size=1):
    logger = autosklearn.util.logging_.get_logger(
        outputdir=log_dir, name="automl_%s" % basename)
    stop = stopwatch.StopWatch()
    stop.start_task(basename)
    stop.start_task("LoadData")

    # == Creating a data object with data and information about it
    logger.debug("======== Reading and converting data ==========")
    # Encoding the labels will be done after the metafeature calculation!
    loaded_data_manager = data_manager.DataManager(basename, input_dir,
                                                   verbose=True,
                                                   encode_labels=False)
    loaded_data_manager_str = str(loaded_data_manager).split("\n")
    for part in loaded_data_manager_str:
        logger.debug(part)

    # == Split dataset and store Data for the ensemble script
    X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(
        loaded_data_manager.data['X_train'], loaded_data_manager.data['Y_train'])
    np.save(os.path.join(tmp_dataset_dir, "true_labels_ensemble.npy"), Y_ensemble)
    del X_train, X_ensemble, Y_train, Y_ensemble

    time_needed_to_load_data = stop.wall_elapsed(basename)
    time_left_after_reading = max(0, time_left_for_this_task -
                                  time_needed_to_load_data)
    logger.info("Remaining time after reading %s %5.2f sec" % (basename, time_left_after_reading))

    stop.stop_task("LoadData")
    # = Create a searchspace
    stop.start_task("CreateSearchSpace")
    searchspace_path = os.path.join(tmp_dataset_dir, "space.pcs")
    config_space = paramsklearn.get_configuration_space(loaded_data_manager.info)

    # Remove configurations we think are unhelpful
    combinations = [("no_preprocessing", "gaussian_nb"),
                    ("sparse_filtering", "gaussian_nb"),
                    ("pca", "gaussian_nb"),
                    ("select_percentile", "gaussian_nb"),
                    ("kitchen_sinks", "gaussian_nb"),
                    ("no_preprocessing", "multinomial_nb"),
                    ("select_percentile", "multinomial_nb"),
                    ("no_preprocessing", "k_nearest_neighbors"),
                    ("pca", "k_nearest_neighbors"),
                    ("random_trees_embedding", "k_nearest_neighbors"),
                    ("pca", "adaboost"),
                    ("no_preprocessing", "adaboost"),
                    ("select_percentile", "adaboost"),
                    ("sparse_filtering", "liblinear"),
                    ("sparse_filtering", "sgd")]
    for combination in combinations:
        try:
            config_space.add_forbidden(ForbiddenAndConjunction(
                ForbiddenEqualsClause(config_space.get_hyperparameter("preprocessor"),
                                      combination[0]),
                ForbiddenEqualsClause(config_space.get_hyperparameter("classifier"),
                                      combination[1])
            ))
        except:
            pass

    sp_string = pcs_parser.write(config_space)
    fh = open(searchspace_path, 'w')
    fh.write(sp_string)
    fh.close()
    logger.debug("Searchspace written to %s" % searchspace_path)
    stop.stop_task("CreateSearchSpace")

    # == Calculate metafeatures
    stop.start_task("CalculateMetafeatures")
    categorical = [True if feat_type.lower() in ["categorical"] else False
                   for feat_type in loaded_data_manager.feat_type]

    if initial_configurations_via_metalearning <= 0:
        ml = None
    elif loaded_data_manager.info["task"].lower() in \
            ["multiclass.classification", "binary.classification"]:
        ml = metalearning.MetaLearning()
        logger.debug("Start calculating metafeatures for %s" %
                     loaded_data_manager.basename)
        ml.calculate_metafeatures_with_labels(loaded_data_manager.data["X_train"],
                                              loaded_data_manager.data["Y_train"],
                                              categorical=categorical,
                                              dataset_name=loaded_data_manager.basename)
    else:
        ml = None
        logger.critical("Metafeatures not calculated")
    stop.stop_task("CalculateMetafeatures")
    logger.debug("Calculating Metafeatures (categorical attributes) took %5.2f" % stop.wall_elapsed("CalculateMetafeatures"))

    stop.start_task("OneHot")
    loaded_data_manager.perform1HotEncoding()
    stop.stop_task("OneHot")

    stop.start_task("CalculateMetafeaturesEncoded")
    if ml is None:
        initial_configurations = []
    elif loaded_data_manager.info["task"].lower() in \
            ["multiclass.classification", "binary.classification"]:
        ml.calculate_metafeatures_encoded_labels(X_train=loaded_data_manager.data["X_train"],
                                                 Y_train=loaded_data_manager.data["Y_train"],
                                                 categorical=[False] * loaded_data_manager.data["X_train"].shape[0],
                                                 dataset_name=loaded_data_manager.basename)

        logger.debug(ml._metafeatures_labels)
        logger.debug(ml._metafeatures_encoded_labels)

        # TODO check that Metafeatures only contain finite numbers!

        stop.start_task("InitialConfigurations")
        initial_configurations = ml.create_metalearning_string_for_smac_call(
            config_space, loaded_data_manager.basename, loaded_data_manager.info[
                'metric'], initial_configurations_via_metalearning)
        stop.stop_task("InitialConfigurations")
        logger.debug("Looking for initial configurations took %5.2fsec" %
                     stop.wall_elapsed("InitialConfigurations"))
    else:
        initial_configurations = []
        logger.critical("Metafeatures encoded not calculated")

    stop.stop_task("CalculateMetafeaturesEncoded")
    logger.debug("Calculating Metafeatures (encoded attributes) took %5.2fsec" %
                 stop.wall_elapsed("CalculateMetafeaturesEncoded"))
    logger.info("Time left for %s after calculating metafeatures: %5.2fsec" %
                (basename, time_left_for_this_task - stop.wall_elapsed(basename)))

    # == Pickle the data manager
    stop.start_task("StoreDatamanager")
    data_manager_path = os.path.join(tmp_dataset_dir, basename + "_Manager.pkl")
    pickle.dump(loaded_data_manager, open(data_manager_path, 'w'), protocol=-1)
    logger.debug("Pickled Datamanager under %s" % data_manager_path)
    stop.stop_task("StoreDatamanager")

    # == RUN SMAC
    stop.start_task("runSmac")
    # = Create an empty instance file
    instance_file = os.path.join(tmp_dataset_dir, "instances.txt")
    fh = open(instance_file, 'w')
    fh.write("holdout")
    fh.close()
    logger.debug("Create instance file %s" % instance_file)

    # = Start SMAC
    dataset = os.path.join(input_dir, basename)
    time_left_for_smac = max(0, time_left_for_this_task - (stop.wall_elapsed(basename)))
    logger.debug("Start SMAC with %5.2fsec time left" % time_left_for_smac)
    proc_smac = \
        submit_process.run_smac(dataset=dataset,
                                tmp_dir=tmp_dataset_dir,
                                searchspace=searchspace_path,
                                instance_file=instance_file,
                                limit=time_left_for_smac,
                                initial_challengers=initial_configurations)
    stop.stop_task("runSmac")

    # == RUN ensemble builder
    stop.start_task("runEnsemble")
    time_left_for_ensembles = max(0, time_left_for_this_task - (stop.wall_elapsed(basename)))
    logger.debug("Start Ensemble with %5.2fsec time left" % time_left_for_ensembles)
    proc_ensembles = \
        submit_process.run_ensemble_builder(tmp_dir=tmp_dataset_dir,
                                            dataset_name=basename,
                                            task_type=loaded_data_manager.info['task'],
                                            metric=loaded_data_manager.info['metric'],
                                            limit=time_left_for_ensembles,
                                            output_dir=output_dir,
                                            ensemble_size=ensemble_size)
    stop.stop_task("runEnsemble")

    queue.put([time_needed_to_load_data, data_manager_path,
               proc_smac, proc_ensembles])
    return
