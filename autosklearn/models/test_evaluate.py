from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor

import time

try:
    import cPickle as pickle
except:
    import pickle

import autosklearn.models.evaluate


class Test_Evaluator(autosklearn.models.evaluate.Evaluator):
    def __init__(self, Datamanager, configuration, with_predictions=False, all_scoring_functions=False, seed=1):

        self.starttime = time.time()

        self.configuration = configuration

        self.X_train = Datamanager.data['X_train']
        self.Y_train = Datamanager.data['Y_train']

        self.X_test = Datamanager.data.get('X_test')
        self.Y_test = Datamanager.data.get('Y_test')

        self.metric = Datamanager.info['metric']
        self.task_type = Datamanager.info['task'].lower()
        self.seed = seed

        self.with_predictions = with_predictions
        self.all_scoring_functions = all_scoring_functions

        if self.task_type == 'regression':
            self.model = ParamSklearnRegressor(configuration, seed)
            self.predict_function = autosklearn.models.evaluate.predict_regression
        else:
            self.model = ParamSklearnClassifier(configuration, seed)
            self.predict_function = autosklearn.models.evaluate.predict_proba

    #override
    def predict(self, train=False):

        if train:
            Y_pred = self.predict_function(self.X_train, self.model, self.task_type)
            score = autosklearn.models.evaluate.calculate_score(solution=self.Y_train, prediction=Y_pred,
                                                               task_type=self.task_type, metric=self.metric,
                                                               all_scoring_functions=self.all_scoring_functions)
        else:
            Y_pred = self.predict_function(self.X_test, self.model, self.task_type)
            score = autosklearn.models.evaluate.calculate_score(solution=self.Y_test, prediction=Y_pred,
                                                               task_type=self.task_type, metric=self.metric,
                                                               all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, "__len__"):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_pred
        return err
