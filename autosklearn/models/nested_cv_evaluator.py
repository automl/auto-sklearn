# -*- encoding: utf-8 -*-
from collections import defaultdict

import numpy as np

import sklearn.utils
from autosklearn.data.split_data import get_CV_fold
from autosklearn.models.evaluator import Evaluator, calculate_score


class NestedCVEvaluator(Evaluator):

    def __init__(self, Datamanager, configuration,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_dir=None,
                 output_y_test=False,
                 inner_cv_folds=5,
                 outer_cv_folds=5,
                 num_run=None):
        super(NestedCVEvaluator, self).__init__(
            Datamanager, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_dir=output_dir,
            output_y_test=output_y_test,
            num_run=num_run)

        self.inner_cv_folds = inner_cv_folds
        self.outer_cv_folds = outer_cv_folds
        self.Y_optimization = None

        self.outer_models = [None] * outer_cv_folds
        self.inner_models = [None] * outer_cv_folds
        self.X_train = Datamanager.data['X_train']
        self.Y_train = Datamanager.data['Y_train']
        for i in range(outer_cv_folds):
            self.inner_models[i] = [None] * inner_cv_folds

        self.outer_indices = [None] * outer_cv_folds
        self.inner_indices = [None] * outer_cv_folds
        for i in range(outer_cv_folds):
            self.inner_indices[i] = [None] * inner_cv_folds

        self.random_state = sklearn.utils.check_random_state(seed)

    def fit(self):
        seed = self.random_state.randint(1000000)
        for outer_fold in range(self.outer_cv_folds):
            # First perform the fit for the outer cross validation
            outer_train_indices, outer_test_indices = \
                get_CV_fold(self.X_train, self.Y_train, fold=outer_fold,
                            folds=self.outer_cv_folds, shuffle=True,
                            random_state=seed)
            self.outer_indices[outer_fold] = ((outer_train_indices,
                                               outer_test_indices))

            model = self.model_class(self.configuration, self.random_state)
            self.outer_models[outer_fold] = model
            self.outer_models[outer_fold].fit(
                self.X_train[outer_train_indices],
                self.Y_train[outer_train_indices])

            # Then perform the fit for the inner cross validation
            for inner_fold in range(self.inner_cv_folds):
                X_train = self.X_train[outer_train_indices]
                Y_train = self.Y_train[outer_train_indices]
                inner_train_indices, inner_test_indices = \
                    get_CV_fold(X_train, Y_train,
                                fold=inner_fold, folds=self.inner_cv_folds,
                                shuffle=True, random_state=seed)
                inner_train_indices = outer_train_indices[inner_train_indices]
                inner_test_indices = outer_train_indices[inner_test_indices]
                X_train = self.X_train[inner_train_indices]
                Y_train = self.Y_train[inner_train_indices]

                self.inner_indices[outer_fold][inner_fold] = \
                    ((inner_train_indices, inner_test_indices))
                model = self.model_class(self.configuration, self.random_state)
                model = model.fit(X_train, Y_train)
                self.inner_models[outer_fold][inner_fold] = model

    def predict(self):
        # First, obtain the predictions for the ensembles, the validation and
        #  the test set!
        outer_scores = defaultdict(list)
        inner_scores = defaultdict(list)
        Y_optimization_pred = [None] * self.outer_cv_folds
        Y_targets = [None] * self.outer_cv_folds
        Y_valid_pred = [None] * self.outer_cv_folds
        Y_test_pred = [None] * self.outer_cv_folds

        for i in range(self.outer_cv_folds):
            train_indices, test_indices = self.outer_indices[i]
            opt_pred = self.predict_function(
                self.X_train[test_indices], self.outer_models[i],
                self.task_type,
                Y_train=self.Y_train[train_indices])

            Y_optimization_pred[i] = opt_pred
            Y_targets[i] = self.Y_train[test_indices]

            if self.X_valid is not None:
                X_valid = self.X_valid.copy()
                valid_pred = self.predict_function(
                    X_valid, self.outer_models[i], self.task_type,
                    Y_train=self.Y_train[train_indices])
                Y_valid_pred[i] = valid_pred

            if self.X_test is not None:
                X_test = self.X_test.copy()
                test_pred = self.predict_function(
                    X_test, self.outer_models[i], self.task_type,
                    Y_train=self.Y_train[train_indices])
                Y_test_pred[i] = test_pred

        # Calculate the outer scores
        for i in range(self.outer_cv_folds):
            scores = calculate_score(
                Y_targets[i], Y_optimization_pred[i], self.task_type,
                self.metric, self.D.info['target_num'],
                all_scoring_functions=self.all_scoring_functions)
            if self.all_scoring_functions:
                for score_name in scores:
                    outer_scores[score_name].append(scores[score_name])
            else:
                outer_scores[self.metric].append(scores)

        Y_optimization_pred = np.concatenate(
            [Y_optimization_pred[i] for i in range(self.outer_cv_folds)
             if Y_optimization_pred[i] is not None])
        Y_targets = np.concatenate([Y_targets[i]
                                    for i in range(self.outer_cv_folds)
                                    if Y_targets[i] is not None])

        if self.X_valid is not None:
            Y_valid_pred = np.array([Y_valid_pred[i]
                                     for i in range(self.outer_cv_folds)
                                     if Y_valid_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_valid_pred.shape) == 3:
                Y_valid_pred = np.nanmean(Y_valid_pred, axis=0)

        if self.X_test is not None:
            Y_test_pred = np.array([Y_test_pred[i]
                                    for i in range(self.outer_cv_folds)
                                    if Y_test_pred[i] is not None])
            # Average the predictions of several models
            if len(Y_test_pred.shape) == 3:
                Y_test_pred = np.nanmean(Y_test_pred, axis=0)

        self.Y_optimization = Y_targets

        # Second, calculate the inner score
        for outer_fold in range(self.outer_cv_folds):
            for inner_fold in range(self.inner_cv_folds):
                inner_train_indices, inner_test_indices = self.inner_indices[
                    outer_fold][inner_fold]
                Y_test = self.Y_train[inner_test_indices]
                X_test = self.X_train[inner_test_indices]
                model = self.inner_models[outer_fold][inner_fold]
                Y_hat = self.predict_function(
                    X_test, model, self.task_type,
                    Y_train=self.Y_train[inner_train_indices])
                scores = calculate_score(
                    Y_test, Y_hat, self.task_type, self.metric,
                    self.D.info['target_num'],
                    all_scoring_functions=self.all_scoring_functions)
                if self.all_scoring_functions:
                    for score_name in scores:
                        inner_scores[score_name].append(scores[score_name])
                else:
                    inner_scores[self.metric].append(scores)

        # Average the scores!
        if self.all_scoring_functions:
            inner_err = {
                key: 1 - np.mean(inner_scores[key])
                for key in inner_scores
            }
            outer_err = {
                'outer:%s' % key: 1 - np.mean(outer_scores[key])
                for key in outer_scores
            }
            inner_err.update(outer_err)
        else:
            inner_err = 1 - np.mean(inner_scores[self.metric])

        if self.with_predictions:
            return inner_err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return inner_err
