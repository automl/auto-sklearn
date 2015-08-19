# -*- encoding: utf-8 -*-
from autosklearn.constants import *
from autosklearn.data.split_data import split_data
from autosklearn.models.evaluator import Evaluator, calculate_score


class HoldoutEvaluator(Evaluator):

    def __init__(self, datamanager, configuration, with_predictions=False,
                 all_scoring_functions=False, seed=1, output_dir=None,
                 output_y_test=False, num_run=None):
        super(HoldoutEvaluator, self).__init__(
            datamanager, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_dir=output_dir,
            output_y_test=output_y_test,
            num_run=num_run)

        classification = datamanager.info['task'] in CLASSIFICATION_TASKS
        self.X_train, self.X_optimization, self.Y_train, self.Y_optimization = \
            split_data(datamanager.data['X_train'],
                       datamanager.data['Y_train'],
                       classification=classification)

        self.model = self.model_class(self.configuration, self.seed)

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

        score = calculate_score(
            self.Y_optimization, Y_optimization_pred, self.task_type,
            self.metric, self.D.info['target_num'],
            all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, '__len__'):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        if self.with_predictions:
            return err, Y_optimization_pred, Y_valid_pred, Y_test_pred
        return err
