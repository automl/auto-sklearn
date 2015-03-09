import numpy as np

from autosklearn.data.split_data import split_data, get_CV_fold
from autosklearn.models.evaluator import Evaluator, calculate_score


class HoldoutEvaluator(Evaluator):
    def __init__(self, Datamanager, configuration, with_predictions=False,
                 all_scoring_functions=False, seed=1, output_dir=None,
                 output_y_test=False, nested_cv_folds=10):
        super(HoldoutEvaluator, self).__init__(Datamanager, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed, output_dir=output_dir,
            output_y_test=output_y_test)

        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
            split_data(Datamanager.data['X_train'], Datamanager.data['Y_train'])

        self.model = self.model_class(self.configuration, self.seed)
        self.nested_cv_folds = nested_cv_folds

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        Y_optimization_pred = self.predict_function(self.X_optimization,
                                                    self.model, self.task_type)
        if self.X_valid is not None:
            Y_valid_pred = self.predict_function(self.X_valid, self.model,
                                                 self.task_type)
        else:
            Y_valid_pred = None
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model,
                                                self.task_type)
        else:
            Y_test_pred = None

        score = calculate_score(self.Y_optimization, Y_optimization_pred,
                                self.task_type, self.metric,
                                all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, "__len__"):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return err

    def nested_fit(self):
        self.models = [None] * self.nested_cv_folds
        self.indices = [None] * self.nested_cv_folds

        for fold in range(self.nested_cv_folds):
            self.partial_nested_fit(fold)

    def nested_predict(self):
        scores = []
        for i, model, indices in zip(range(self.nested_cv_folds), self.models, self.indices):
            scores.append(self.partial_nested_predict(i))
        scores = np.array(scores)

        if self.all_scoring_functions:
            err = np.mean(scores, axis=1)
        else:
            err = np.mean(scores)

        return err

    def partial_nested_fit(self, fold):
        model = self.model_class(self.configuration, self.seed)

        train_indices, test_indices = \
            get_CV_fold(self.X_train, self.Y_train, fold=fold,
                        folds=self.nested_cv_folds)

        if hasattr(self, "models"):
            self.indices[fold] = ((train_indices, test_indices))

            self.models[fold] = model
            self.models[fold].fit(self.X_train[train_indices],
                                  self.Y_train[train_indices])
        else:
            self.partial_indices = ((train_indices, test_indices))
            self.partial_model = model
            self.partial_model.fit(self.X_train[train_indices],
                                   self.Y_train[train_indices])

    def partial_nested_predict(self, fold):
        if hasattr(self, "models"):
            model = self.models[fold]
            train_indices, test_indices = self.indices[fold]
        else:
            model = self.partial_model
            train_indices, test_indices = self.partial_indices

        opt_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type)

        score = calculate_score(self.Y_train[test_indices], opt_pred,
                                self.task_type, self.metric,
                                all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, "__len__"):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score
        return err