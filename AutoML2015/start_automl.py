import cPickle
import os
import time

import numpy as np

from data.data_io import vprint
from data import cdata_manager as data_manager
from util import split_data
from models import autosklearn
from util import submit_process

from HPOlibConfigSpace.converters import pcs_parser


def start_automl_on_dataset(basename, input_dir, tmp_dataset_dir, output_dir,
                            time_left_for_this_task, queue):
    start = time.time()
    verbose = True
    # == Creating a data object with data and information about it
    vprint(verbose,  "======== Reading and converting data ==========")
    loaded_data_manager = data_manager.DataManager(basename, input_dir, verbose=verbose)
    print loaded_data_manager

    # == Split dataset and store Data, Datamanager
    X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(loaded_data_manager.data['X_train'], loaded_data_manager.data['Y_train'])
    del X_train, X_ensemble, Y_train
    np.save(os.path.join(tmp_dataset_dir, "true_labels_ensemble.npy"), Y_ensemble)
    data_manager_path = os.path.join(tmp_dataset_dir, basename + "_Manager.pkl")
    cPickle.dump(loaded_data_manager, open(data_manager_path, 'w'), protocol=-1)

    stop = time.time()
    time_needed_to_load_data = stop - start
    time_left_after_reading = max(0, time_left_for_this_task - time_needed_to_load_data)
    vprint(verbose, "Remaining time after reading data %5.2f sec" % time_left_after_reading)

    # == RUN SMAC
    # = Create an empty instance file
    instance_file = os.path.join(tmp_dataset_dir, "instances.txt")
    fh = open(instance_file, 'w')
    fh.write(os.path.join(input_dir, basename))
    fh.close()

    # = Create a searchspace
    searchspace = os.path.join(tmp_dataset_dir, "space.pcs")
    sp = autosklearn.get_configuration_space(loaded_data_manager.info)
    sp_string = pcs_parser.write(sp)
    fh = open(searchspace, 'w')
    fh.write(sp_string)
    fh.close()

    # = Start SMAC
    stop = time.time()
    time_left_for_smac = max(0, time_left_for_this_task - (stop - start))
    proc_smac = \
        submit_process.run_smac(tmp_dir=tmp_dataset_dir,
                                searchspace=searchspace,
                                instance_file=instance_file,
                                limit=time_left_for_smac)
    pid_smac = proc_smac.pid

    # == RUN ensemble builder
    stop = time.time()
    time_left_for_ensembles = max(0, time_left_for_this_task - (stop - start))
    proc_ensemble = \
        submit_process.run_ensemble_builder(tmp_dir=tmp_dataset_dir,
                                            dataset_name=basename,
                                            task_type=loaded_data_manager.info['task'],
                                            metric=loaded_data_manager.info['metric'],
                                            limit=time_left_for_ensembles,
                                            output_dir=output_dir)

    pid_ensembles = proc_ensemble.pid
    queue.put([time_needed_to_load_data, data_manager_path, pid_smac, pid_ensembles])
    return