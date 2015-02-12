'''
Created on Feb 4, 2015

@author: Aaron Klein
'''
import os
import glob
import numpy as np

from data.data_manager import DataManager
from ensemble_script import weighted_ensemble, ensemble_prediction
from util.get_dataset_info import getInfoFromFile
from ensembles_tweakathon import load_predictions, load_predictions_of_best, load_predictions_of_nbest
from ensembles.greedy_search import GreedySearch
from ensembles import stacking
from models import evaluate

from sklearn.cross_validation import train_test_split


def acq_fkt(X_train, X_test, Y_train, Y_test, task_type, metric):
    init_weights = np.ones([X_train.shape[0]]) / float(X_train.shape[0])

    weights = weighted_ensemble(X_train, Y_train, task_type, metric, init_weights)
    ensemble_pred = ensemble_prediction(X_test, weights)
    score = evaluate.calculate_score(Y_test, ensemble_pred, task_type, metric)
    return score


def acq_fkt_stacking(X_train, X_test, Y_train, Y_test, task_type, metric):
    n_points = X_test.shape[1]
    n_classes = X_test.shape[2]

    # Convert data set for stacking, i.e. add softmax indices and flatten array
    X_train, Y_train = stacking.convert_data_set(X_train, Y_train)
    X_test = stacking.convert_predictions(X_test)

    ensemble_pred = stacking.stacking_logistic_regression(X_train, Y_train, X_test)

    ensemble_pred = np.reshape(ensemble_pred, (n_points, n_classes))

    score = evaluate.calculate_score(Y_test, ensemble_pred, task_type, metric)
    return score


def greedy_search(X, y, task_type, metric):
    split = int(X.shape[1] * 0.7)
    X_train = X[:, :split, :]
    X_test = X[:, split:, :]
    y_train = y[:split]
    y_test = y[split:]

    gs = GreedySearch(acq_fkt)
    trajectory = gs.search(X_train, X_test, y_train, y_test, task_type, metric)
    print "Trajectory: "
    print trajectory
    order = gs.get_order()
    print "Complete order: "
    print order
    # Check were the trajectory starts to overfit
    perf = 0
    for i in range(0, trajectory.shape[0]):
        if perf <= trajectory[i]:
            perf = trajectory[i]
        else:
            order = order[:i]
            break
    print "Number of models: " + str(order.shape[0])
    print order
    return order.astype(int)


def main(dataset, path="/home/feurerm/mhome/projects/automl_competition_2015/tweakathon/"):

    print "Use data set: " + str(dataset)

    dirs = glob.glob(os.path.join(path, dataset, "smac_2_08_00-master_*"))
    output_dir = "./predictions_greedy_search_tweakathon/"
    data_dir = "/data/aad/automl_data"

    try:
        os.mkdir(output_dir)
    except:
        pass

    info = getInfoFromFile(data_dir, dataset)

    print "Load labels from " + str(os.path.join(path, dataset, dataset + ".npy"))
    true_labels = np.load(os.path.join(path, dataset, dataset + ".npy"))

    #predictions, predictions_valid, predictions_test = load_predictions_of_best(dirs, true_labels, info['task'], info['metric'], load_all_predictions=True)
    predictions, predictions_valid, predictions_test, indices, dirs_nbest = load_predictions_of_nbest(dirs, 200, true_labels, info['task'], info['metric'], load_all_predictions=True)
    #predictions, predictions_valid, predictions_test = load_predictions(dirs, load_all_predictions=True)
    print "number of models: " + str(predictions.shape[0])

#     D = DataManager(dataset, data_dir, verbose=True)
#     predictions_valid, predictions_test = train_models_on_complete_data(indices_nbest, dirs_nbest,
#                                                                         D.data['X_train'], D.data['Y_train'],
#                                                                         D.data['X_valid'], D.data['X_test'], info['task'], predictions_valid, predictions_test)
    print "Start greedy search"
    indices_models = greedy_search(predictions, true_labels, info['task'], info['metric'])

    predictions = predictions[indices_models]
    predictions_valid = predictions_valid[indices_models]
    predictions_test = predictions_test[indices_models]

    #TODO: "Re-train models with the whole data set"

    print "Re-start optimization"
    weights = np.ones([predictions.shape[0]]) / float(predictions.shape[0])
    weights = weighted_ensemble(predictions, true_labels, info['task'], info['metric'], weights)
    print "Best weights found by CMA-ES: " + str(weights)

    print "Compute ensembles predictions for valid data"
    Y_valid = ensemble_prediction(predictions_valid, weights)

    print "Compute ensemble predictions for test data"
    Y_test = ensemble_prediction(predictions_test, weights)

    print "Save predictions in: " + output_dir

    np.savetxt(output_dir + dataset + "_valid_000.predict", Y_valid, delimiter=' ')
    np.savetxt(output_dir + dataset + "_test_000.predict", Y_test, delimiter=' ')

if __name__ == '__main__':
    #dataset = ["adult", "digits", "newsgroups", "dorothea", "cadata"]
    dataset = ["adult", "digits", "newsgroups", "cadata"]
    for d in dataset:
        main(d)
