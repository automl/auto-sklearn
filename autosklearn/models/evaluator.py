import abc
import os
import time

import numpy as np

from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor

from autosklearn.scores import libscores
from autosklearn.data.split_data import split_data


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


class Evaluator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, Datamanager, configuration, with_predictions=False,
                 all_scoring_functions=False, seed=1, output_dir=None,
                 output_y_test=False):

        self.starttime = time.time()

        self.configuration = configuration
        self.D = Datamanager

        self.X_valid = Datamanager.data.get('X_valid')
        self.X_test = Datamanager.data.get('X_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task'].lower()
        self.seed = seed

        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir

        self.output_y_test = output_y_test
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type == 'regression':
            self.model_class = ParamSklearnRegressor
            self.predict_function = predict_regression
        else:
            self.model_class = ParamSklearnClassifier
            self.predict_function = predict_proba

    @abc.abstractmethod
    def fit(self):
        pass

    # This function does everything necessary after the fitting is done:
    #        predicting
    #        saving the files for the ensembles_statistics
    #        generate output for SMAC
    # We use it as the signal handler so we can recycle the code for the normal usecase and when the runsolver kills us here :)
    def finish_up(self):
        try:
            self.duration = time.time() - self.starttime
            result, additional_run_info = self.file_output()
            print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % ("SAT", abs(self.duration), result, self.seed, additional_run_info)
        except:
            self.duration = time.time() - self.starttime
            import sys
            e = sys.exc_info()[0]
            print e
            print
            print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % (
                "TIMEOUT", abs(self.duration), 1.0, self.seed,
                "No results were produced! Probably the training was not "
                "finished and no valid model was generated!")

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def nested_fit(self):
        pass

    @abc.abstractmethod
    def nested_predict(self):
        pass

    @abc.abstractmethod
    def partial_nested_fit(self, fold):
        pass

    @abc.abstractmethod
    def partial_nested_predict(self, fold):
        pass
    
    def file_output(self):
        errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = self.predict()
        num_run = str(get_new_run_num())
        pred_dump_name_template = os.path.join(self.output_dir,
            "predictions_%s", self.D.basename + '_predictions_%s_' +
            num_run + '.npy')

        if self.output_y_test:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            with open(os.path.join(self.output_dir, "y_optimization.npy"), "w") as fh:
                pickle.dump(self.Y_optimization, fh, -1)

        ensemble_output_dir = os.path.join(self.output_dir, "predictions_ensemble")
        if not os.path.exists(ensemble_output_dir):
            os.makedirs(ensemble_output_dir)
        with open(pred_dump_name_template % ("ensemble", "ensemble"), "w") as fh:
            pickle.dump(Y_optimization_pred, fh, -1)

        valid_output_dir = os.path.join(self.output_dir, "predictions_valid")
        if not os.path.exists(valid_output_dir):
            os.makedirs(valid_output_dir)
        with open(pred_dump_name_template % ("valid", "valid"), "w") as fh:
            pickle.dump(Y_valid_pred, fh, -1)

        test_output_dir = os.path.join(self.output_dir, "predictions_test")
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        with open(pred_dump_name_template % ("test", "test"), "w") as fh:
            pickle.dump(Y_test_pred, fh, -1)

        self.duration = time.time() - self.starttime
        err = errs[self.D.info['metric']]
        additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
        additional_run_info += ";" + "duration: " + str(self.duration)
        additional_run_info += ";" + "num_run:" + num_run
        return err, additional_run_info
