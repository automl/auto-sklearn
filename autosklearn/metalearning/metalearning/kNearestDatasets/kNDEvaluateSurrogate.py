from __future__ import print_function
import logging
import os
import time

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import scipy.stats

import pyMetaLearn.directory_manager
from pyMetaLearn.metalearning.meta_base import MetaBase
import HPOlib.benchmark_util as benchmark_util
from pyMetaLearn.metalearning.kNearestDatasets.kND import LearnedDistanceRF

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """For a given problem train the metric function and return its loss
    value. Arguments:
      * task_files_list
      * experiment_files_list
      * metalearning_directory


    Sample call: python kNDEvaluateSurrogate.py --task_files_list
    /mhome/feurerm/thesis/experiments/AutoSklearn/metalearning_experiments/2014_09_10_test/tasks.txt
    --experiments_list /mhome/feurerm/thesis/experiments/AutoSklearn
    /metalearning_experiments/2014_09_10_test/experiments.txt
    --metalearning_directory /mhome/feurerm/thesis/experiments/AutoSklearn/ --params -random_state 5
    """

    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    os.chdir(args['metalearning_directory'])
    pyMetaLearn.directory_manager.set_local_directory(
        args['metalearning_directory'])

    with open(args["task_files_list"]) as fh:
        task_files_list = fh.readlines()
    with open(args["experiments_list"]) as fh:
        experiments_list = fh.readlines()

    if 'keep_configurations' in args:
        keep_configurations = args['keep_configurations']
        keep_configurations = keep_configurations.split(',')
        keep_configurations = tuple(
            [tuple(kc.split('=')) for kc in keep_configurations])
    else:
        keep_configurations = None

    meta_base = MetaBase(task_files_list, experiments_list, keep_configurations)
    metafeatures = meta_base.get_all_train_metafeatures_as_pandas()
    runs = meta_base.get_all_runs()
    split_masks = dict()
    training = dict()

    # This can print the best hyperparameters of every dataset
    # for dataset in runs:
    # print dataset, sorted(runs[dataset], key=lambda t: t.result)[0]

    for i, name in enumerate(runs):
        runs[name].sort()
        rs = np.random.RandomState(i*37)
        ones = np.ones((200,))
        zeros = np.zeros((len(runs[name]) - len(ones),))
        numbers = np.append(ones, zeros)
        rs.shuffle(numbers)
        split_masks[name] = numbers
        training[name] = [run for j, run in enumerate(runs[name]) if numbers[j]]

    rf = LearnedDistanceRF(**params)
    filled_runs = rf._apply_surrogates(metafeatures, training)


    # Now sort the arrays so we can compare it to the ground truth in run
    for name in runs:
        filled_runs[name].sort()
        print(len(filled_runs[name]), len(runs[name]))
        offset = 0
        a1 = []
        a2 = []
        for i in range(len(filled_runs[name])):
            while True:
                if filled_runs[name][i].params == runs[name][i+offset].params:
                    a1.append(filled_runs[name][i].result)
                    a2.append(runs[name][i+offset].result)
                    break
                else:
                    offset += 1
        a1 = pd.Series(a1)
        a2 = pd.Series(a2)
        a1.fillna(1, inplace=True)
        a2.fillna(1, inplace=True)
        print(sklearn.metrics.mean_absolute_error(a1, a2), \
              sklearn.metrics.mean_squared_error(a1, a2), \
              np.sqrt(sklearn.metrics.mean_squared_error(a1, a2)), \
              scipy.stats.spearmanr(a1, a2)[0])





