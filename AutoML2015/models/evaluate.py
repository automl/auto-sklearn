'''
Created on Dec 18, 2014

@author: Aaron Klein
'''

import numpy as np

from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.autosklearn_regression import AutoSklearnRegressor

from data.data_converter import convert_to_bin
from scores import libscores
from util.split_data import split_data
import time
import os

try:
    import cPickle as pickle
except:
    import pickle



def predict_proba(X, model, task_type):
    Y_pred = model.predict_proba(X)

    if task_type == "multilabel.classification":
        Y_pred = np.hstack(
            [Y_pred[i][:, 1].reshape((-1, 1))
             for i in range(len(Y_pred))])

    elif task_type == "binary.classification":
        if len(Y_pred.shape) != 1:
            Y_pred = Y_pred[:, 1].reshape(-1, 1)

    return Y_pred


def predict_regression(X, model, task_type):
    Y_pred = model.predict(X)

    if len(Y_pred.shape) == 1:
        Y_pred = Y_pred.reshape((-1, 1))

    return Y_pred


def calculate_score(solution, prediction, task_type, metric,
                    all_scoring_functions=False):
    if task_type == "multiclass.classification":
        solution_binary = np.zeros((prediction.shape))
        for i in range(solution_binary.shape[0]):
            label = solution[i]
            solution_binary[i, label] = 1
        solution = solution_binary

    elif task_type in ["binary.classification", "regression"]:
        if len(solution.shape) == 1:
            solution = solution.reshape((-1, 1))

    scoring_func = getattr(libscores, metric)

    if all_scoring_functions:
        score = dict()
        if task_type == "regression":
            cprediction = libscores.sanitize_array(prediction)
            score["a_metric"] = libscores.a_metric(solution, cprediction,
                                                   task=task_type)
            score["r2_metric"] = libscores.r2_metric(solution, cprediction,
                                                     task=task_type)
        else:
            csolution, cprediction = libscores.normalize_array(
                solution, prediction)
            score["bac_metric"] = libscores.bac_metric(csolution, cprediction,
                                                       task=task_type)
            score["auc_metric"] = libscores.auc_metric(csolution, cprediction,
                                                       task=task_type)
            score["f1_metric"] = libscores.f1_metric(csolution, cprediction,
                                                     task=task_type)
            score["pac_metric"] = libscores.pac_metric(csolution, cprediction,
                                                       task=task_type)

    else:
        if task_type == "regression":
            cprediction = libscores.sanitize_array(prediction)
            score = scoring_func(solution, cprediction, task=task_type)
        else:
            csolution, cprediction = libscores.normalize_array(
                solution, prediction)
            score = scoring_func(csolution, cprediction, task=task_type)
    return score




def get_new_run_num():
    counter_file = os.path.join(os.getcwd(), "num_run")
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as fh:
            fh.write("0")
        return 0
    else:
        with open(counter_file, "r") as fh:
            num = int(fh.read())
        num += 1
        with open(counter_file, "w") as fh:
            fh.write(str(num))
        return num


class evaluator:
    def __init__(self, Datamanager, configuration, with_predictions=False, all_scoring_functions=False, splitting_function=split_data, seed=1):

        self.starttime = time.time()

        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
        splitting_function(Datamanager.data['X_train'], Datamanager.data['Y_train'])

        self.X_valid = Datamanager.data.get('X_valid')
        self.X_test = Datamanager.data.get('X_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task'].lower()
        self.seed = seed

        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type == 'regression':
            self.model = AutoSklearnRegressor(configuration, seed)
            self.predict_function = predict_regression
        else:
            self.model = AutoSklearnClassifier(configuration, seed)
            self.predict_function = predict_proba

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    # This function does everything necessary after the fitting is done:
    #        predicting
    #        saving the files for the ensembles
    #        generate output for SMAC
    # We use it as the signal handler so we can recycle the code for the normal usecase and when the runsolver kills us here :)
    def finish_up(self):
        try:
            self.duration = time.time() - self.starttime
            result, additional_run_info = self.file_output()
            print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % ("SAT", abs(self.duration), result, self.seed, additional_run_info);
        except:
            import sys
            e = sys.exc_info()[0]
            print e
            print "No results were produced! Probably the training was not finished, and no valid model was generated!"

    def predict(self):

        Y_optimization_pred = self.predict_function(self.X_optimization, self.model, self.task_type)
        if self.X_valid is not None:
            Y_valid_pred = self.predict_function(self.X_valid, self.model, self.task_type)
        else:
            Y_valid_pred = None
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model, self.task_type)
        else:
            Y_test_pred = None

        score = calculate_score(self.Y_optimization, Y_optimization_pred, self.task_type, self.metric, all_scoring_functions=self.all_scoring_functions)
        
        if hasattr(score, "__len__"):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return err
    
    def file_output(self):
        errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = self.predict()
        pred_dump_name_template = os.path.join(self.output_dir, "predictions_%s", self.basename + '_predictions_%s_' + str(get_new_run_num()) + '.npy')

        ensemble_output_dir = os.path.join(self.output_dir, "predictions_ensemble")
        if not os.path.exists(ensemble_output_dir):
            os.mkdir(ensemble_output_dir)
        with open(pred_dump_name_template % ("ensemble", "ensemble"), "w") as fh:
            pickle.dump(Y_optimization_pred, fh, -1)

        valid_output_dir = os.path.join(self.output_dir, "predictions_valid")
        if not os.path.exists(valid_output_dir):
            os.mkdir(valid_output_dir)
        with open(pred_dump_name_template % ("valid", "valid"), "w") as fh:
            pickle.dump(Y_valid_pred, fh, -1)

        test_output_dir = os.path.join(self.output_dir, "predictions_test")
        if not os.path.exists(test_output_dir):
            os.mkdir(test_output_dir)
        with open(pred_dump_name_template % ("test", "test"), "w") as fh:
            pickle.dump(Y_test_pred, fh, -1)

        err = errs[self.D.info['metric']]
        print errs
        import sys
        sys.stdout.flush()
        additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
        additional_run_info += ";" + "duration: " + str(self.duration)
        additional_run_info += ";" + "prediction_files_template: " + \
            pred_dump_name_template
        return err, additional_run_info
        
"""
def evaluate(Datamanager, configuration, with_predictions=False,
        all_scoring_functions=False, splitting_function=split_data, seed=1):
            
    X_train, X_optimization, Y_train, Y_optimization = \
        splitting_function(Datamanager.data['X_train'], Datamanager.data['Y_train'])
    X_valid = Datamanager.data.get('X_valid')
    X_test = Datamanager.data.get('X_test')

    metric = Datamanager.info['metric']
    task_type = Datamanager.info['task'].lower()

   if task_type == 'regression':
        model = AutoSklearnRegressor(configuration, seed)
    else:
        model = AutoSklearnClassifier(configuration, seed)

    model.fit(X_train, Y_train)

    if task_type == 'regression':
        predict_function = predict_regression
    else:
        predict_function = predict_proba

    Y_optimization_pred = predict_function(X_optimization, model, task_type)
    if X_valid is not None:
        Y_valid_pred = predict_function(X_valid, model, task_type)
    else:
        Y_valid_pred = None
    if X_test is not None:
        Y_test_pred = predict_function(X_test, model, task_type)
    else:
        Y_test_pred = None

    score = calculate_score(Y_optimization, Y_optimization_pred,
                            task_type, metric,
                            all_scoring_functions=all_scoring_functions)
    if hasattr(score, "__len__"):
        err = {key: 1 - score[key] for key in score}
    else:
        err = 1 - score

    if with_predictions:
        return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
    else:
        return err
"""
