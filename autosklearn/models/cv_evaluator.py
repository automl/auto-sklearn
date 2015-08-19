# -*- encoding: utf-8 -*-
import numpy as np

from autosklearn.data.split_data import get_CV_fold
from autosklearn.models.evaluator import Evaluator, calculate_score


class CVEvaluator(Evaluator):

    def __init__(self, Datamanager, configuration,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_dir=None,
                 output_y_test=False,
                 cv_folds=10,
                 num_run=None):
        super(CVEvaluator, self).__init__(
            Datamanager, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_dir=output_dir,
            output_y_test=output_y_test,
            num_run=num_run)

        self.cv_folds = cv_folds
        self.X_train = self.D.data['X_train']
        self.Y_train = self.D.data['Y_train']
        self.Y_optimization = None

        self.models = [None] * cv_folds
        self.indices = [None] * cv_folds

    def fit(self):
        for fold in range(self.cv_folds):
            self.partial_fit(fold)

    def partial_fit(self, fold):
        model = self.model_class(self.configuration, self.seed)

        train_indices, test_indices = \
            get_CV_fold(self.X_train, self.Y_train, fold=fold,
                        folds=self.cv_folds, shuffle=True,
                        random_state=self.seed)

        self.indices[fold] = ((train_indices, test_indices))

        self.models[fold] = model
        self.models[fold].fit(self.X_train[train_indices],
                              self.Y_train[train_indices])

    def predict(self):
        Y_optimization_pred = [None] * self.cv_folds
        Y_targets = [None] * self.cv_folds
        Y_valid_pred = [None] * self.cv_folds
        Y_test_pred = [None] * self.cv_folds

        for i in range(self.cv_folds):
            # To support prediction when only partial_fit was called
            if self.models[i] is None:
                continue

            train_indices, test_indices = self.indices[i]
            opt_pred = self.predict_function(self.X_train[test_indices],
                                             self.models[i], self.task_type,
                                             self.Y_train[train_indices])

            Y_optimization_pred[i] = opt_pred
            Y_targets[i] = self.Y_train[test_indices]

            if self.X_valid is not None:
                X_valid = self.X_valid.copy()
                valid_pred = self.predict_function(X_valid, self.models[i],
                                                   self.task_type,
                                                   self.Y_train[train_indices])
                Y_valid_pred[i] = valid_pred

            if self.X_test is not None:
                X_test = self.X_test.copy()
                test_pred = self.predict_function(X_test, self.models[i],
                                                  self.task_type,
                                                  self.Y_train[train_indices])
                Y_test_pred[i] = test_pred

        Y_optimization_pred = np.concatenate(
            [Y_optimization_pred[i] for i in range(self.cv_folds)
             if Y_optimization_pred[i] is not None])
        Y_targets = np.concatenate([Y_targets[i] for i in range(self.cv_folds)
                                    if Y_targets[i] is not None])

        if self.X_valid is not None:
            Y_valid_pred = np.array([Y_valid_pred[i]
                                     for i in range(self.cv_folds)
                                     if Y_valid_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_valid_pred.shape) == 3:
                Y_valid_pred = np.nanmean(Y_valid_pred, axis=0)

        if self.X_test is not None:
            Y_test_pred = np.array([Y_test_pred[i]
                                    for i in range(self.cv_folds)
                                    if Y_test_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_test_pred.shape) == 3:
                Y_test_pred = np.nanmean(Y_test_pred, axis=0)

        self.Y_optimization = Y_targets
        score = calculate_score(
            Y_targets, Y_optimization_pred, self.task_type, self.metric,
            self.D.info['target_num'],
            all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, '__len__'):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return err
