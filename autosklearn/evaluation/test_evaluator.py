# -*- encoding: utf-8 -*-
from smac.tae.execute_ta_run import StatusType

from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.metrics import calculate_score


__all__ = [
    'eval_t',
    'TestEvaluator'
]


class TestEvaluator(AbstractEvaluator):

    def __init__(self, backend, queue, metric,
                 configuration=None,
                 all_scoring_functions=False,
                 seed=1,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None):
        super(TestEvaluator, self).__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_hat_optimization=False,
            num_run=-1,
            subsample=None,
            include=include,
            exclude=exclude,
            disable_file_output= disable_file_output,
            init_params=init_params
        )
        self.configuration = configuration

        self.X_train = self.datamanager.data['X_train']
        self.Y_train = self.datamanager.data['Y_train']

        self.X_test = self.datamanager.data.get('X_test')
        self.Y_test = self.datamanager.data.get('Y_test')

        self.model = self._get_model()

    def fit_predict_and_loss(self):
        self._fit_and_suppress_warnings(self.model, self.X_train, self.Y_train)
        loss, Y_pred, _, _ =  self.predict_and_loss()
        self.finish_up(
            loss=loss,
            train_pred=None,
            opt_pred=Y_pred,
            valid_pred=None,
            test_pred=None,
            file_output=False,
            final_call=True,
            additional_run_info=None,
        )

    def predict_and_loss(self, train=False):

        if train:
            Y_pred = self.predict_function(self.X_train, self.model,
                                           self.task_type, self.Y_train)
            score = calculate_score(
                solution=self.Y_train,
                prediction=Y_pred,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=self.all_scoring_functions)
        else:
            Y_pred = self.predict_function(self.X_test, self.model,
                                           self.task_type, self.Y_train)
            score = calculate_score(
                solution=self.Y_test,
                prediction=Y_pred,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=self.all_scoring_functions)

        if hasattr(score, '__len__'):
            err = {key: 1 - score[key] for key in score}
        else:
            err = 1 - score

        return err, Y_pred, None, None


# create closure for evaluating an algorithm
# Has a stupid name so nosetests doesn't regard it as a test
def eval_t(queue, config, backend, metric, seed, num_run, instance,
           all_scoring_functions, output_y_hat_optimization, include,
           exclude, disable_file_output, init_params=None):
    evaluator = TestEvaluator(configuration=config,
                              backend=backend, metric=metric, seed=seed,
                              queue=queue,
                              all_scoring_functions=all_scoring_functions,
                              include=include, exclude=exclude,
                              disable_file_output=disable_file_output,
                              init_params=init_params)

    evaluator.fit_predict_and_loss()


