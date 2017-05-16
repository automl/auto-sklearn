from collections import defaultdict
import itertools
import logging
import os
import Queue
import time

import numpy as np
import pandas as pd
import sklearn
import scipy.stats

from autosklearn.metalearning.metalearning.meta_base import MetaBase
import HPOlib.benchmark_util as benchmark_util
from autosklearn.metalearning.metalearning.kNearestDatasets.kND import \
    LearnedDistanceRF

logger = logging.getLogger(__name__)


# ###############################################################################
# Stuff for offline hyperparameter optimization of the Distance RF

def _validate_rf_without_one_dataset(X, Y, rf, task_id):
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, task_id)
    predictions = rf.model.predict(X_valid)
    rho = scipy.stats.kendalltau(Y_valid, predictions)[0]
    mae = sklearn.metrics.mean_absolute_error(predictions, Y_valid)
    mse = sklearn.metrics.mean_squared_error(predictions, Y_valid)
    return mae, mse, rho


def train_rf_without_one_dataset(X, Y, rf, task_id):
    # Pay attention, this is not for general sklearn models, but for adapted
    # models...
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, task_id)
    rf.model.fit(X_train, Y_train)
    return rf


def split_for_loo(X, Y, task_id):
    train = []
    valid = []
    for cross in X.index:
        if str(task_id) in cross:
            valid.append(cross)
        else:
            train.append(cross)

    X_train = X.loc[train].values
    Y_train = Y.loc[train].values.reshape((-1,))
    X_valid = X.loc[valid].values
    Y_valid = Y.loc[valid].values.reshape((-1,))
    return X_train, Y_train, X_valid, Y_valid


# TODO: this file has too many tasks, move the optimization of the metric
# function and the forward selection to some different files,
# maybe generalize these things to work for other models as well...
if __name__ == "__main__":
    """For a given problem train the metric function and return its loss
    value. Arguments:
      * task_files_list
      * experiment_files_list
      * metalearning_directory

    You can also enable forward selection by adding '--forward_selection True'
    You can also enable embedded feature selection by adding '--embedded_selection True'
    You can add '--keep_configurations -preprocessing=None,-classifier=LibSVM
    Sample call: python kND.py --task_files_list /home/feurerm/thesis/experiments/experiment/2014_06_01_AutoSklearn_metalearning/tasks.txt
    --experiments_list /home/feurerm/thesis/experiments/experiment/2014_06_01_AutoSklearn_metalearning/experiments_fold0.txt
    --metalearning_directory /home/feurerm/thesis/experiments/experiment --params -random_state 5
    """
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    os.chdir(args['metalearning_directory'])
    #pyMetaLearn.directory_manager.set_local_directory(
    #    args['metalearning_directory'])

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

    # This can print the best hyperparameters of every dataset
    # for dataset in runs:
    # print dataset, sorted(runs[dataset], key=lambda t: t.result)[0]

    rf = LearnedDistanceRF(**params)
    X, Y = rf._create_dataset(metafeatures, runs)
    import cPickle

    with open("test.pkl", "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)

    print("Metafeatures", metafeatures.shape)
    print("X", X.shape, np.isfinite(X).all().all())
    print("Y", Y.shape, np.isfinite(Y).all())

    metafeature_sets = Queue.Queue()
    if 'forward_selection' in args:
        used_metafeatures = []
        metafeature_performance = []
        print("Starting forward selection ",)
        i = 0
        for m1, m2 in itertools.combinations(metafeatures.columns, 2):
            metafeature_sets.put(pd.Index([m1, m2]))
            i += 1
        print("with %d metafeature combinations" % i)
    elif 'embedded_selection' in args:
        metafeature_performance = []
        metafeature_sets.put(metafeatures.columns)
    else:
        metafeature_sets.put(metafeatures.columns)

    while not metafeature_sets.empty():
        metafeature_set = metafeature_sets.get()
        metafeature_ranks = defaultdict(float)
        loo_mae = []
        loo_rho = []
        loo_mse = []

        print("###############################################################")
        print("New iteration of FS with:")
        print(metafeature_set)
        print("Dataset Mae MSE Rho")
        # Leave one out CV
        for idx in range(metafeatures.shape[0]):
            leave_out_dataset = metafeatures.index[idx]
            if 'forward_selection' not in args:
                print(leave_out_dataset,)

            columns = np.hstack(("0_" + metafeature_set,
                                 "1_" + metafeature_set))
            X_ = X.loc[:, columns]
            rf = train_rf_without_one_dataset(X_, Y, rf, leave_out_dataset)
            mae, mse, rho = _validate_rf_without_one_dataset(X_, Y, rf,
                                                             leave_out_dataset)
            if 'forward_selection' not in args:
                print(mae, mse, rho)

            loo_mae.append(mae)
            loo_rho.append(rho)
            loo_mse.append(mse)
            mf_importances = [(rf.model.feature_importances_[i], X_.columns[i])
                              for i in range(X_.shape[1])]
            mf_importances.sort()
            mf_importances.reverse()
            for rank, item in enumerate(mf_importances):
                score, mf_name = item
                metafeature_ranks[mf_name] += float(rank)

        mae = np.mean(loo_mae)
        mae_std = np.std(loo_mae)
        mse = np.mean(loo_mse)
        mse_std = np.mean(loo_mse)
        rho = np.mean(loo_rho)
        rho_std = np.std(loo_rho)

        mean_ranks = [
            (metafeature_ranks[mf_name] / metafeatures.shape[0], mf_name)
            for mf_name in X.columns]
        mean_ranks.sort()

        # TODO: save only validate-best runs!
        print("MAE", mae, mae_std)
        print("MSE", mse, mse_std)
        print("Mean tau", rho, rho_std)
        duration = time.time() - starttime

        if 'forward_selection' in args:
            metafeature_performance.append((mse, metafeature_set))
            # TODO: this can also be sorted in a pareto-optimal way...

            if metafeature_sets.empty():
                if len(used_metafeatures) == 10:
                    break

                print("#######################################################")
                print("#######################################################")
                print("Adding a new feature to the feature set")
                metafeature_performance.sort()
                print(metafeature_performance)
                used_metafeatures = metafeature_performance[0][1]
                for metafeature in metafeatures.columns:
                    if metafeature in used_metafeatures:
                        continue
                    # I don't know if indexes are copied
                    tmp = [uaie for uaie in used_metafeatures]
                    tmp.append(metafeature)
                    metafeature_sets.put(pd.Index(tmp))
                metafeature_performance = []

        elif 'embedded_selection' in args:
            if len(metafeature_set) <= 2:
                break

            # Remove a metafeature; elements are (average rank, name);
            # only take the name from index two on
            # because the metafeature is preceeded by the index of the
            # dataset which is either 0_ or 1_
            remove = mean_ranks[-1][1][2:]
            print("Going to remove", remove)
            keep = pd.Index([mf_name for mf_name in metafeature_set if
                             mf_name != remove])
            print("I will keep", keep)
            metafeature_sets.put(keep)

        else:
            for rank in mean_ranks:
                print(rank)

    if 'forward_selection' in args:
        metafeature_performance.sort()
        print(metafeature_performance)
        mse = metafeature_performance[0][0]
    print("Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
          ("SAT", abs(duration), mse, -1, str(__file__)))