import abc
import os
import time

import lockfile
import numpy as np

from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor

from autosklearn.scores import libscores
from autosklearn.constants import *


try:
    import cPickle as pickle
except:
    import pickle


def calculate_score(solution, prediction, task_type, metric, num_classes,
                    all_scoring_functions=False):
    if task_type == MULTICLASS_CLASSIFICATION:
        solution_binary = np.zeros((prediction.shape[0], num_classes))
        for i in range(solution_binary.shape[0]):
            label = solution[i]
            solution_binary[i, label] = 1
        solution = solution_binary

    elif task_type in [BINARY_CLASSIFICATION, REGRESSION]:
        if len(solution.shape) == 1:
            solution = solution.reshape((-1, 1))

    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    scoring_func = getattr(libscores, metric)
    if solution.shape != prediction.shape:
        raise ValueError("Solution shape %s != prediction shape %s" %
                         (solution.shape, prediction.shape))

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
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
            score["acc_metric"] = libscores.acc_metric(csolution, cprediction,
                                                       task=task_type)

    else:
        if task_type in REGRESSION_TASKS:
            cprediction = libscores.sanitize_array(prediction)
            score = scoring_func(solution, cprediction, task=task_type)
        else:
            csolution, cprediction = libscores.normalize_array(
                solution, prediction)
            score = scoring_func(csolution, cprediction, task=task_type)
    return score


def get_new_run_num():
    seed = os.environ.get("AUTOSKLEARN_SEED")
    counter_file = "num_run"
    if seed is not None:
        counter_file = counter_file + ("_%s" % seed)
    counter_file = os.path.join(os.getcwd(), counter_file)
    lock = lockfile.LockFile(counter_file)
    with lock:
        if not os.path.exists(counter_file):
            with open(counter_file, "w") as fh:
                fh.write("0")
            num = 0
        else:
            with open(counter_file, "r") as fh:
                num = int(fh.read())
            num += 1
            with open(counter_file, "w") as fh:
                fh.write(str(num).zfill(4))

    return num


class Evaluator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, Datamanager, configuration, with_predictions=False,
                 all_scoring_functions=False, seed=1, output_dir=None,
                 output_y_test=False, num_run=None):

        self.starttime = time.time()

        self.configuration = configuration
        self.D = Datamanager

        self.X_valid = Datamanager.data.get('X_valid')
        self.X_test = Datamanager.data.get('X_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task']
        self.seed = seed

        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir

        self.output_y_test = output_y_test
        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type in REGRESSION_TASKS:
            self.model_class = ParamSklearnRegressor
            self.predict_function = self.predict_regression
        else:
            self.model_class = ParamSklearnClassifier
            self.predict_function = self.predict_proba

        if num_run is None:
            num_run = get_new_run_num()
        self.num_run = num_run

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
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
            import traceback
            print traceback.format_exc()
            print
            print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % (
                "TIMEOUT", abs(self.duration), 1.0, self.seed,
                "No results were produced! Probably the training was not "
                "finished and no valid model was generated!")
    
    def file_output(self):
        seed = os.environ.get("AUTOSKLEARN_SEED")
        errs, Y_optimization_pred, Y_valid_pred, Y_test_pred = self.predict()
        num_run = str(self.num_run).zfill(5)
        pred_dump_name_template = os.path.join(self.output_dir,
            "predictions_%s_%s", self.D.basename + '_predictions_%s_' +
            num_run + '.npy')

        if self.output_y_test:
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass
            y_test_filename = os.path.join(self.output_dir,
                                           "y_optimization.npy")
            with lockfile.LockFile(y_test_filename + '.lock'):
                with open(y_test_filename, "w") as fh:
                    pickle.dump(self.Y_optimization.astype(np.float32), fh, -1)

        ensemble_output_dir = os.path.join(self.output_dir,
                                           "predictions_ensemble_%s" % seed)
        if not os.path.exists(ensemble_output_dir):
            os.makedirs(ensemble_output_dir)
        with open(pred_dump_name_template % ("ensemble", seed, "ensemble"), "w") as fh:
            pickle.dump(Y_optimization_pred.astype(np.float32), fh, -1)

        if Y_valid_pred is not None:
            valid_output_dir = os.path.join(self.output_dir,
                                            "predictions_valid_%s" % seed)
            if not os.path.exists(valid_output_dir):
                os.makedirs(valid_output_dir)
            with open(pred_dump_name_template % ("valid", seed, "valid"), "w") as fh:
                pickle.dump(Y_valid_pred.astype(np.float32), fh, -1)

        if Y_test_pred is not None:
            test_output_dir = os.path.join(self.output_dir,
                                           "predictions_test_%s" % seed)
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            with open(pred_dump_name_template % ("test", seed, "test"), "w") as fh:
                pickle.dump(Y_test_pred.astype(np.float32), fh, -1)

        self.duration = time.time() - self.starttime
        err = errs[self.D.info['metric']]
        additional_run_info = ";".join(["%s: %s" % (metric, value)
                                    for metric, value in errs.items()])
        additional_run_info += ";" + "duration: " + str(self.duration)
        additional_run_info += ";" + "num_run:" + num_run
        # print "Saved predictions with shapes %s, %s, %s for num_run %s" % \
        #       (Y_optimization_pred.shape, Y_valid_pred.shape,
        #        Y_test_pred.shape, num_run)
        return err, additional_run_info

    def predict_proba(self, X, model, task_type, Y_train=None):
        Y_pred = model.predict_proba(X, batch_size=1000)

        if task_type == MULTILABEL_CLASSIFICATION:
            Y_pred = np.hstack(
                [Y_pred[i][:, -1].reshape((-1, 1))
                 for i in range(len(Y_pred))])

        elif task_type == BINARY_CLASSIFICATION:
            if len(Y_pred.shape) != 1:
                Y_pred = Y_pred[:, 1].reshape(-1, 1)

        elif task_type == MULTICLASS_CLASSIFICATION:
            pass

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def predict_regression(self, X, model, task_type, Y_train=None):
        Y_pred = model.predict(X, batch_size=1000)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction, Y_train):
        num_classes = self.D.info["target_num"]

        if self.task_type == MULTICLASS_CLASSIFICATION and \
                        prediction.shape[1] < num_classes:
            classes = list(np.unique(self.D.data["Y_train"]))
            if num_classes == prediction.shape[1]:
                return prediction

            if Y_train is not None:
                classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes))
            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
