from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor

import numpy as np

from ..data.split_data import split_data, get_CV_fold
from .evaluate import Evaluator, calculate_score


class CVEvaluator(Evaluator):
    def __init__(self, Datamanager, configuration, with_predictions=False,
                 all_scoring_functions=False, splitting_function=split_data,
                 seed=1, output_dir=None, output_y_test=False, cv_folds=10):
        super(CVEvaluator, self).__init__(Datamanager, configuration,
                                          with_predictions=with_predictions,
                                          all_scoring_functions=all_scoring_functions,
                                          splitting_function=splitting_function,
                                          seed=seed, output_dir=output_dir,
                                          output_y_test=output_y_test)

        self.cv_folds = cv_folds
        self.X_train = self.D.data['X_train']
        self.Y_train = self.D.data['Y_train']

    def fit(self):
        self.models = []
        self.indices = []
        for i in range(self.cv_folds):
            if self.task_type == 'regression':
                model = ParamSklearnRegressor(self.configuration,
                                              self.seed)
            else:
                model = ParamSklearnClassifier(self.configuration,
                                               self.seed)

            train_indices, test_indices = \
                get_CV_fold(self.X_train, self.Y_train, fold=i, folds=self.cv_folds)

            self.indices.append((train_indices, test_indices))

            self.models.append(model.fit(self.X_train[train_indices],
                                         self.Y_train[train_indices]))

    def predict(self):
        for i in range(self.cv_folds):
            train_indices, test_indices = self.indices[i]

            opt_pred = self.predict_function(self.X_train[test_indices],
                                             self.models[i], self.task_type)
            if i == 0:
                Y_optimization_pred = opt_pred
                Y_targets = self.Y_train[test_indices]
            else:
                Y_optimization_pred = np.concatenate(
                    (Y_optimization_pred, opt_pred))
                Y_targets = np.concatenate(
                    (Y_targets, self.Y_train[test_indices]))

            if self.X_valid is not None:
                valid_pred = self.predict_function(self.X_valid,
                                                     self.models[i],
                                                     self.task_type)
                if i == 0:
                    Y_valid_pred = [valid_pred]
                else:
                    Y_valid_pred.append(valid_pred)

            else:
                Y_valid_pred = None

            if self.X_test is not None:
                test_pred = self.predict_function(self.X_test, self.models[i],
                                                    self.task_type)
                if i == 0:
                    Y_test_pred = [test_pred]
                else:
                    Y_test_pred.append(test_pred)
            else:
                Y_test_pred = None

        if self.X_valid is not None:
            Y_valid_pred = np.mean(np.array(Y_valid_pred), axis=0)
        if self.X_test is not None:
            Y_test_pred = np.mean(np.array(Y_test_pred), axis=0)

        score = calculate_score(Y_targets, Y_optimization_pred,
                                self.task_type, self.metric,
                                all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, "__len__"):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return err