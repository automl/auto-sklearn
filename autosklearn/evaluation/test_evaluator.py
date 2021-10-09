# -*- encoding: utf-8 -*-
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration

import numpy as np

from smac.tae import StatusType

from autosklearn.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    _fit_and_suppress_warnings,
)
from autosklearn.metrics import calculate_loss, Scorer
from autosklearn.util.backend import Backend


__all__ = [
    'eval_t',
    'TestEvaluator'
]


class TestEvaluator(AbstractEvaluator):

    def __init__(
        self,
        backend: Backend,
        queue: multiprocessing.Queue,
        metric: Scorer,
        port: Optional[int],
        configuration: Optional[Union[int, Configuration]] = None,
        scoring_functions: Optional[List[Scorer]] = None,
        seed: int = 1,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        disable_file_output: bool = False,
        init_params: Optional[Dict[str, Any]] = None,
    ):
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

    def fit_predict_and_loss(self) -> None:
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

    def predict_and_loss(
        self, train: bool = False
    ) -> Tuple[Union[Dict[str, float], float], np.array, Any, Any]:

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
def eval_t(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: Dict[str, Any],
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
) -> None:
    evaluator = TestEvaluator(configuration=config,
                              backend=backend, metric=metric, seed=seed,
                              port=port,
                              queue=queue,
                              scoring_functions=scoring_functions,
                              include=include, exclude=exclude,
                              disable_file_output=disable_file_output,
                              init_params=init_params)

    evaluator.fit_predict_and_loss()
