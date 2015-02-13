import cPickle
import os
import time

import numpy as np

from data.data_io import vprint
from data import data_manager as data_manager
from metalearning import metalearning
from models import autosklearn
from util import split_data
from util import submit_process

from HPOlibConfigSpace.converters import pcs_parser


def start_automl_on_dataset(basename, input_dir, tmp_dataset_dir, output_dir,
                            time_left_for_this_task, queue):
    start = time.time()
    verbose = True
    # == Creating a data object with data and information about it
    vprint(verbose,  "======== Reading and converting data ==========")
    # Encoding the labels will be done after the metafeature calculation!
    loaded_data_manager = data_manager.DataManager(basename, input_dir,
                                                   verbose=verbose,
                                                   encode_labels=False)
    print loaded_data_manager

    # == Split dataset and store Data for the ensemble script
    X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(
        loaded_data_manager.data['X_train'], loaded_data_manager.data['Y_train'])
    np.save(os.path.join(tmp_dataset_dir, "true_labels_ensemble.npy"), Y_ensemble)
    del X_train, X_ensemble, Y_train, Y_ensemble

    stop = time.time()
    time_needed_to_load_data = stop - start
    time_left_after_reading = max(0, time_left_for_this_task -
                                  time_needed_to_load_data)
    vprint(verbose, "Remaining time after reading data %5.2f sec" % time_left_after_reading)

    # = Create a searchspace
    searchspace_path = os.path.join(tmp_dataset_dir, "space.pcs")
    config_space = autosklearn.get_configuration_space(loaded_data_manager.info)
    sp_string = pcs_parser.write(config_space)
    fh = open(searchspace_path, 'w')
    fh.write(sp_string)
    fh.close()

    # == Calculate metafeatures
    categorical = [True if feat_type.lower() in ["categorical"] else False
                   for feat_type in loaded_data_manager.feat_type]

    if loaded_data_manager.info["task"].lower() not in \
            ["multilabel.classification", "regression"] and \
            not loaded_data_manager.info["is_sparse"]:
        ml = metalearning.MetaLearning()
        metafeatures_start_time = time.time()
        vprint(verbose, "Start calculating metafeatures for %s" %
               loaded_data_manager.basename)
        ml.calculate_metafeatures_with_labels(loaded_data_manager.data["X_train"],
                                              loaded_data_manager.data["Y_train"],
                                              categorical=categorical,
                                              dataset_name=loaded_data_manager.basename)

    loaded_data_manager.perform1HotEncoding()

    if loaded_data_manager.info["task"].lower() not in \
            ["multilabel.classification", "regression"] and \
            not loaded_data_manager.info["is_sparse"]:
        ml.calculate_metafeatures_encoded_labels(loaded_data_manager.data["X_train"],
                                                 loaded_data_manager.data["Y_train"],
            categorical=[False]*loaded_data_manager.data["X_train"].shape[0],
            dataset_name=loaded_data_manager.basename)
        metafeatures_end_time = time.time()
        metafeature_calculation_time = metafeatures_end_time - metafeatures_start_time
        vprint(verbose, "Done calculationg metafeatures for %s, took %5.2f "
                        "seconds." % (loaded_data_manager.basename,
                                      metafeature_calculation_time))
        time_left_after_metafeatures = max(0, time_left_for_this_task -
                                           (metafeatures_end_time - start))
        vprint(verbose,
           "Remaining time after calculating the metafeatures for %s %5.2f "
           "sec" % (loaded_data_manager.basename, time_left_after_metafeatures))

        vprint(verbose, ml._metafeatures_labels)
        vprint(verbose, ml._metafeatures_encoded_labels)

        # TODO check that Metafeatures only contain finite numbers!

        vprint(verbose, "Starting to look for initial configurations for %s." % loaded_data_manager.basename)
        initial_configurations_start_time = time.time()
        initial_configurations = ml.create_metalearning_string_for_smac_call(
            config_space, loaded_data_manager.basename, loaded_data_manager.info[
                'metric'])
        initial_configurations_end_time = time.time()
        vprint(verbose, "Calculating the initial configurations for %s took "
                        "%5.2f seconds" % (loaded_data_manager.basename,
                                           initial_configurations_end_time -
                                           initial_configurations_start_time))

        time_left_after_initial_configurations = max(0,
            time_left_for_this_task - (initial_configurations_end_time - start))

        vprint(verbose,
               "Remaining time after finding the initial configurations for %s "
               "%5.2f sec" % (loaded_data_manager.basename,
                              time_left_after_initial_configurations))

    else:
        initial_configurations = []

    # == Pickle the data manager
    data_manager_path = os.path.join(tmp_dataset_dir, basename + "_Manager.pkl")
    cPickle.dump(loaded_data_manager, open(data_manager_path, 'w'), protocol=-1)

    # == RUN SMAC
    # = Create an empty instance file
    instance_file = os.path.join(tmp_dataset_dir, "instances.txt")
    fh = open(instance_file, 'w')
    fh.write(os.path.join(input_dir, basename))
    fh.close()

    # = Start SMAC
    stop = time.time()
    time_left_for_smac = max(0, time_left_for_this_task - (stop - start))
    pid_smac = \
        submit_process.run_smac(tmp_dir=tmp_dataset_dir,
                                searchspace=searchspace_path,
                                instance_file=instance_file,
                                limit=time_left_for_smac,
                                initial_challengers=initial_configurations)


    # == RUN ensemble builder
    stop = time.time()
    time_left_for_ensembles = max(0, time_left_for_this_task - (stop - start))
    pid_ensembles = \
        submit_process.run_ensemble_builder(tmp_dir=tmp_dataset_dir,
                                            dataset_name=basename,
                                            task_type=loaded_data_manager.info['task'],
                                            metric=loaded_data_manager.info['metric'],
                                            limit=time_left_for_ensembles,
                                            output_dir=output_dir)

    queue.put([time_needed_to_load_data, data_manager_path, pid_smac, pid_ensembles])
    return
