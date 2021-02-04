# -*- encoding: utf-8 -*-
from smac.tae import StatusType

from autosklearn.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    _fit_and_suppress_warnings,
)
from autosklearn.metrics import calculate_loss


__all__ = [
    'eval_t',
    'TestEvaluator'
]


class TestEvaluator(AbstractEvaluator):

    def __init__(self, backend, queue, metric,
                 port,
                 configuration=None,
                 scoring_functions=None,
                 seed=1,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None):
        super(TestEvaluator, self).__init__(
            backend=backend,
            queue=queue,
            port=port,
            configuration=configuration,
            metric=metric,
            scoring_functions=scoring_functions,
            seed=seed,
            output_y_hat_optimization=False,
            num_run=-1,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params
        )
        self.configuration = configuration

        self.X_train = self.datamanager.data['X_train']
        self.Y_train = self.datamanager.data['Y_train']

        self.X_test = self.datamanager.data.get('X_test')
        self.Y_test = self.datamanager.data.get('Y_test')

        self.model = self._get_model()

    def fit_predict_and_loss(self):
        _fit_and_suppress_warnings(self.logger, self.model, self.X_train, self.Y_train)
        loss, Y_pred, _, _ = self.predict_and_loss()
        self.finish_up(
            loss=loss,
            train_loss=None,
            opt_pred=Y_pred,
            valid_pred=None,
            test_pred=None,
            file_output=False,
            final_call=True,
            additional_run_info=None,
            status=StatusType.SUCCESS,
        )

    def predict_and_loss(self, train=False):

        if train:
            Y_pred = self.predict_function(self.X_train, self.model,
                                           self.task_type, self.Y_train)
            err = calculate_loss(
                solution=self.Y_train,
                prediction=Y_pred,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=self.scoring_functions)
        else:
            Y_pred = self.predict_function(self.X_test, self.model,
                                           self.task_type, self.Y_train)
            err = calculate_loss(
                solution=self.Y_test,
                prediction=Y_pred,
                task_type=self.task_type,
                metric=self.metric,
                scoring_functions=self.scoring_functions)

        return err, Y_pred, None, None


# create closure for evaluating an algorithm
# Has a stupid name so pytest doesn't regard it as a test
def eval_t(queue, config, backend, metric, seed, num_run, instance,
           scoring_functions, output_y_hat_optimization, include,
           exclude, disable_file_output, port, init_params=None, budget_type=None,
           budget=None):
    evaluator = TestEvaluator(configuration=config,
                              backend=backend, metric=metric, seed=seed,
                              port=port,
                              queue=queue,
                              scoring_functions=scoring_functions,
                              include=include, exclude=exclude,
                              disable_file_output=disable_file_output,
                              init_params=init_params)

    evaluator.fit_predict_and_loss()
