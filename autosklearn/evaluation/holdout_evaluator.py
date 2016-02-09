# -*- encoding: utf-8 -*-
from __future__ import print_function

from autosklearn.constants import *
from autosklearn.evaluation.resampling import split_data
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator


__all__ = [
    'HoldoutEvaluator'
]


class HoldoutEvaluator(AbstractEvaluator):

    def __init__(self, datamanager, output_dir,
                 configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 num_run=None):
        super(HoldoutEvaluator, self).__init__(
            datamanager, output_dir, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run)

        classification = datamanager.info['task'] in CLASSIFICATION_TASKS
        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
            split_data(datamanager.data['X_train'],
                       datamanager.data['Y_train'],
                       classification=classification)

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def iterative_fit(self):
        Xt, fit_params = self.model.pre_transform(self.X_train, self.Y_train)
        if not self.model.estimator_supports_iterative_fit():
            print("Model does not support iterative_fit(), reverting to " \
                "regular fit().")

            self.model.fit_estimator(Xt, self.Y_train, **fit_params)
            return

        n_iter = 1
        while not self.model.configuration_fully_fitted():
            self.model.iterative_fit(Xt, self.Y_train, n_iter=n_iter,
                                     **fit_params)
            self.file_output()
            n_iter += 2

    def predict(self):
        Y_optimization_pred = self.predict_function(self.X_optimization,
                                                    self.model, self.task_type,
                                                    self.Y_train)
        if self.X_valid is not None:
            Y_valid_pred = self.predict_function(self.X_valid, self.model,
                                                 self.task_type,
                                                 self.Y_train)
        else:
            Y_valid_pred = None
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model,
                                                self.task_type,
                                                self.Y_train)
        else:
            Y_test_pred = None

        return Y_optimization_pred, Y_valid_pred, Y_test_pred

