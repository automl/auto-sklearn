from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import copy
import json
import logging
import multiprocessing
import warnings

import numpy as np
import pandas
import pandas as pd
import scipy.sparse
from ConfigSpace import Configuration
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    PredefinedSplit,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits
from smac.tae import StatusType, TAEAbortException

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import (
    CLASSIFICATION_TASKS,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION_TASKS,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
from autosklearn.evaluation.abstract_evaluator import (
    TYPE_ADDITIONAL_INFO,
    AbstractEvaluator,
    _fit_and_suppress_warnings,
)
from autosklearn.evaluation.splitter import (
    CustomStratifiedKFold,
    CustomStratifiedShuffleSplit,
)
from autosklearn.metrics import Scorer
from autosklearn.pipeline.base import PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import (
    IterativeComponent,
    ThirdPartyComponents,
)
from autosklearn.util.logging_ import PicklableClientLogger

__all__ = [
    "TrainEvaluator",
    "eval_holdout",
    "eval_iterative_holdout",
    "eval_cv",
    "eval_partial_cv",
    "eval_partial_cv_iterative",
]


def _get_y_array(y: SUPPORTED_TARGET_TYPES, task_type: int) -> SUPPORTED_TARGET_TYPES:
    if task_type in CLASSIFICATION_TASKS and task_type != MULTILABEL_CLASSIFICATION:
        return y.ravel()
    else:
        return y


T = TypeVar("T", SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES)


def select(data: T, indices: np.ndarray) -> T:
    """Select into some data by indices"""
    return data.iloc[indices] if hasattr(data, "iloc") else data[indices]


def subsample_indices(
    train_indices: List[int],
    subsample: Optional[float],
    task_type: int,
    Y_train: SUPPORTED_TARGET_TYPES,
) -> List[int]:

    if not isinstance(subsample, float):
        raise ValueError(
            "Subsample must be of type float, but is of type %s" % type(subsample)
        )
    elif subsample > 1:
        raise ValueError("Subsample must not be larger than 1, but is %f" % subsample)

    if subsample is not None and 0 <= subsample < 1:
        # Only subsample if there are more indices given to this method than
        # required to subsample because otherwise scikit-learn will complain

        if task_type in CLASSIFICATION_TASKS and task_type != MULTILABEL_CLASSIFICATION:
            stratify: Optional[SUPPORTED_TARGET_TYPES] = select(Y_train, train_indices)
        else:
            stratify = None

        indices = np.arange(len(train_indices))
        cv_indices_train, _ = train_test_split(
            indices,
            stratify=stratify,
            train_size=subsample,
            random_state=1,
            shuffle=True,
        )
        train_indices = train_indices[cv_indices_train]
        return train_indices

    return train_indices


def _fit_with_budget(
    X_train: SUPPORTED_FEAT_TYPES,
    Y_train: SUPPORTED_TARGET_TYPES,
    budget: float,
    budget_type: Optional[str],
    logger: Union[logging.Logger, PicklableClientLogger],
    model: BaseEstimator,
    train_indices: List[int],
    task_type: int,
) -> None:
    if budget_type == "iterations" or (
        budget_type == "mixed" and model.estimator_supports_iterative_fit()
    ):
        X = select(X_train, train_indices)
        y = select(Y_train, train_indices)

        if model.estimator_supports_iterative_fit():
            budget_factor = model.get_max_iter()
            Xt, fit_params = model.fit_transformer(X, y)

            n_iter = int(np.ceil(budget / 100 * budget_factor))
            model.iterative_fit(Xt, y, n_iter=n_iter, refit=True, **fit_params)
        else:
            _fit_and_suppress_warnings(logger, model, X, y)

    elif budget_type == "subsample" or (
        budget_type == "mixed" and not model.estimator_supports_iterative_fit()
    ):
        subsample = budget / 100
        train_indices_subset = subsample_indices(
            train_indices,
            subsample,
            task_type,
            Y_train,
        )
        X = select(X_train, train_indices_subset)
        y = select(Y_train, train_indices_subset)
        _fit_and_suppress_warnings(logger, model, X, y)

    else:
        raise ValueError(budget_type)


def concat_data(
    data: List[Any], num_cv_folds: int
) -> Union[np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix]:
    if isinstance(data[0], np.ndarray):
        return np.concatenate(
            [data[i] for i in range(num_cv_folds) if data[i] is not None]
        )
    elif isinstance(data[0], scipy.sparse.spmatrix):
        return scipy.sparse.vstack(
            [data[i] for i in range(num_cv_folds) if data[i] is not None]
        )
    elif isinstance(data[0], pd.DataFrame):
        return pd.concat(
            [data[i] for i in range(num_cv_folds) if data[i] is not None],
            axis=0,
        )
    else:
        raise ValueError(f"Unknown datatype {type(data[0])}")


class TrainEvaluator(AbstractEvaluator):
    def __init__(
        self,
        backend: Backend,
        queue: multiprocessing.Queue,
        metrics: Sequence[Scorer],
        additional_components: Dict[str, ThirdPartyComponents],
        port: Optional[int],
        configuration: Optional[Union[int, Configuration]] = None,
        scoring_functions: Optional[List[Scorer]] = None,
        seed: int = 1,
        output_y_hat_optimization: bool = True,
        resampling_strategy: Optional[
            Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit]
        ] = None,
        resampling_strategy_args: Optional[
            Dict[str, Optional[Union[float, int, str]]]
        ] = None,
        num_run: Optional[int] = None,
        budget: Optional[float] = None,
        budget_type: Optional[str] = None,
        keep_models: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        disable_file_output: bool = False,
        init_params: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            backend=backend,
            queue=queue,
            port=port,
            configuration=configuration,
            metrics=metrics,
            additional_components=additional_components,
            scoring_functions=scoring_functions,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
        )

        self.feat_type = self.backend.load_datamanager().feat_type
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            self.resampling_strategy_args = {}
        else:
            self.resampling_strategy_args = resampling_strategy_args
        self.splitter = self.get_splitter(self.datamanager)
        self.num_cv_folds = self.splitter.get_n_splits(
            groups=self.resampling_strategy_args.get("groups")
        )
        self.X_train = self.datamanager.data["X_train"]
        self.Y_train = self.datamanager.data["Y_train"]
        self.X_targets = [None] * self.num_cv_folds
        self.Y_targets = [None] * self.num_cv_folds
        self.Y_train_targets = np.ones(self.Y_train.shape) * np.NaN
        self.models = [None] * self.num_cv_folds
        self.indices: List[Optional[Tuple[List[int], List[int]]]] = [
            None
        ] * self.num_cv_folds

        # Necessary for full CV. Makes full CV not write predictions if only
        # a subset of folds is evaluated but time is up. Complicated, because
        #  code must also work for partial CV, where we want exactly the
        # opposite.
        self.partial = True
        self.keep_models = keep_models

    def fit_predict_and_loss(self, iterative: bool = False) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout (both iterative and non-iterative)
        """
        # Define beforehand for mypy
        additional_run_info: Optional[TYPE_ADDITIONAL_INFO] = None

        if iterative:
            if self.num_cv_folds == 1:

                for train_split, test_split in self.splitter.split(
                    self.X_train,
                    self.Y_train,
                    groups=self.resampling_strategy_args.get("groups"),
                ):
                    self.X_optimization = select(self.X_train, test_split)
                    self.Y_optimization = select(self.Y_train, test_split)
                    self.Y_actual_train = select(self.Y_train, train_split)
                    self._partial_fit_and_predict_iterative(
                        0,
                        train_indices=train_split,
                        test_indices=test_split,
                        add_model_to_self=True,
                    )
            else:

                # Test if the model allows for an iterative fit, if not,
                # call this method again without the iterative argument
                model = self._get_model(self.feat_type)
                if not model.estimator_supports_iterative_fit():
                    self.fit_predict_and_loss(iterative=False)
                    return

                self.partial = False

                converged = [False] * self.num_cv_folds

                Y_train_pred = [None] * self.num_cv_folds
                Y_optimization_pred = [None] * self.num_cv_folds
                Y_test_pred = [None] * self.num_cv_folds
                train_splits = [None] * self.num_cv_folds

                self.models = [
                    self._get_model(self.feat_type) for i in range(self.num_cv_folds)
                ]
                iterations = [1] * self.num_cv_folds
                total_n_iterations = [0] * self.num_cv_folds
                # model.estimator_supports_iterative_fit -> true
                # After the if above, we know estimator support iterative fit
                model_max_iter = [
                    cast(IterativeComponent, model).get_max_iter()
                    for model in self.models
                ]

                if self.budget_type in ["iterations", "mixed"] and self.budget is None:
                    raise ValueError(
                        f"When budget type is {self.budget_type} the budget "
                        "can not be None"
                    )

                if (
                    self.budget_type in ["iterations", "mixed"]
                    and cast(float, self.budget) > 0
                ):
                    max_n_iter_budget = int(
                        np.ceil(cast(float, self.budget) / 100 * model_max_iter[0])
                    )
                    max_iter = min(model_max_iter[0], max_n_iter_budget)
                else:
                    max_iter = model_max_iter[0]

                models_current_iters = [0] * self.num_cv_folds

                Xt_array = [None] * self.num_cv_folds
                fit_params_array = [
                    {}
                ] * self.num_cv_folds  # type: List[Dict[str, Any]]

                y = _get_y_array(self.Y_train, self.task_type)

                # stores train loss(es) of each fold.
                train_losses = [dict()] * self.num_cv_folds
                # used as weights when averaging train losses.
                train_fold_weights = [np.NaN] * self.num_cv_folds
                # stores opt (validation) loss of each fold.
                opt_losses = [np.NaN] * self.num_cv_folds
                # weights for opt_losses.
                opt_fold_weights = [np.NaN] * self.num_cv_folds

                while not all(converged):

                    splitter = self.get_splitter(self.datamanager)

                    for i, (train_indices, test_indices) in enumerate(
                        splitter.split(
                            self.X_train,
                            y,
                            groups=self.resampling_strategy_args.get("groups"),
                        )
                    ):
                        if converged[i]:
                            continue

                        model = self.models[i]

                        if iterations[i] == 1:
                            self.Y_train_targets[train_indices] = select(
                                self.Y_train, train_indices
                            )
                            self.X_targets[i] = select(self.X_train, test_indices)
                            self.Y_targets[i] = select(self.Y_train, test_indices)

                            # Note: Be careful moving these into variables, caused a
                            # headache when trying to debug why things were breaking
                            Xt, fit_params = model.fit_transformer(
                                select(self.X_train, train_indices),
                                select(self.Y_train, train_indices),
                            )
                            Xt_array[i] = Xt
                            fit_params_array[i] = fit_params

                        n_iter = int(2 ** iterations[i] / 2) if iterations[i] > 1 else 2
                        total_n_iterations[i] = total_n_iterations[i] + n_iter

                        model.iterative_fit(
                            Xt_array[i],
                            select(self.Y_train, train_indices),
                            n_iter=n_iter,
                            **fit_params_array[i],
                        )

                        (train_pred, opt_pred, test_pred) = self._predict(
                            model,
                            train_indices=train_indices,
                            test_indices=test_indices,
                        )

                        Y_train_pred[i] = train_pred
                        Y_optimization_pred[i] = opt_pred
                        Y_test_pred[i] = test_pred
                        train_splits[i] = train_indices

                        train_loss = self._loss(
                            select(self.Y_train, train_indices),
                            train_pred,
                            X_data=Xt_array[i],
                        )
                        train_losses[i] = train_loss
                        # Number of training data points for this fold.
                        # Used for weighting the average.
                        train_fold_weights[i] = len(train_indices)

                        # Compute validation loss of this fold and store it.
                        optimization_loss = self._loss(
                            self.Y_targets[i], opt_pred, X_data=self.X_targets[i]
                        )
                        opt_losses[i] = optimization_loss
                        # number of optimization data points for this fold.
                        # Used for weighting the average.
                        opt_fold_weights[i] = len(test_indices)

                        models_current_iters[i] = model.get_current_iter()

                        if (
                            model.configuration_fully_fitted()
                            or models_current_iters[i] >= max_iter
                        ):
                            converged[i] = True

                        iterations[i] = iterations[i] + 1

                    # Compute weights of each fold based on the number of samples
                    # in each fold.
                    train_fold_weights_percentage = [
                        w / sum(train_fold_weights) for w in train_fold_weights
                    ]
                    opt_fold_weights_percentage = [
                        w / sum(opt_fold_weights) for w in opt_fold_weights
                    ]

                    train_loss = {
                        metric.name: np.average(
                            [
                                train_losses[i][str(metric)]
                                for i in range(self.num_cv_folds)
                            ],
                            weights=train_fold_weights_percentage,
                        )
                        for metric in self.metrics
                    }

                    # if all_scoring_function is true, return a dict of opt_loss.
                    # Otherwise, return a scalar.
                    opt_loss = {}
                    for metric in opt_losses[0].keys():
                        opt_loss[metric] = np.average(
                            [opt_losses[i][metric] for i in range(self.num_cv_folds)],
                            weights=opt_fold_weights_percentage,
                        )

                    X_targets = self.X_targets
                    Y_targets = self.Y_targets
                    Y_train_targets = self.Y_train_targets

                    Y_optimization_pred_concat = concat_data(
                        Y_optimization_pred, num_cv_folds=self.num_cv_folds
                    )
                    X_targets = concat_data(X_targets, num_cv_folds=self.num_cv_folds)
                    Y_targets = concat_data(Y_targets, num_cv_folds=self.num_cv_folds)

                    if self.X_test is not None:
                        Y_test_preds = np.array(
                            [
                                Y_test_pred[i]
                                for i in range(self.num_cv_folds)
                                if Y_test_pred[i] is not None
                            ]
                        )
                        # Average the predictions of several models
                        if len(Y_test_preds.shape) == 3:
                            Y_test_preds = np.nanmean(Y_test_preds, axis=0)
                    else:
                        Y_test_preds = None

                    self.X_optimization = X_targets
                    self.Y_optimization = Y_targets
                    self.Y_actual_train = Y_train_targets

                    self.model = self._get_model(self.feat_type)
                    status = StatusType.DONOTADVANCE
                    if any(
                        [
                            model_current_iter == max_iter
                            for model_current_iter in models_current_iters
                        ]
                    ):
                        status = StatusType.SUCCESS
                    self.finish_up(
                        loss=opt_loss,
                        train_loss=train_loss,
                        opt_pred=Y_optimization_pred_concat,
                        test_pred=Y_test_preds,
                        additional_run_info=additional_run_info,
                        file_output=True,
                        final_call=all(converged),
                        status=status,
                    )

        else:

            self.partial = False

            Y_train_pred = [None] * self.num_cv_folds
            Y_optimization_pred = [None] * self.num_cv_folds
            Y_test_pred = [None] * self.num_cv_folds
            train_splits = [None] * self.num_cv_folds

            y = _get_y_array(self.Y_train, self.task_type)

            train_losses = []  # stores train loss of each fold.
            train_fold_weights = []  # used as weights when averaging train losses.
            opt_losses = []  # stores opt (validation) loss of each fold.
            opt_fold_weights = []  # weights for opt_losses.

            # TODO: mention that no additional run info is possible in this
            # case! -> maybe remove full CV from the train evaluator anyway and
            # make the user implement this!
            for i, (train_split, test_split) in enumerate(
                self.splitter.split(
                    self.X_train, y, groups=self.resampling_strategy_args.get("groups")
                )
            ):

                # TODO add check that split is actually an integer array,
                # not a boolean array (to allow indexed assignement of
                # training data later).

                if self.budget_type is None:
                    (
                        train_pred,
                        opt_pred,
                        test_pred,
                        additional_run_info,
                    ) = self._partial_fit_and_predict_standard(
                        i,
                        train_indices=train_split,
                        test_indices=test_split,
                        add_model_to_self=self.num_cv_folds == 1,
                    )
                else:
                    (
                        train_pred,
                        opt_pred,
                        test_pred,
                        additional_run_info,
                    ) = self._partial_fit_and_predict_budget(
                        i,
                        train_indices=train_split,
                        test_indices=test_split,
                        add_model_to_self=self.num_cv_folds == 1,
                    )

                if (
                    additional_run_info is not None
                    and len(additional_run_info) > 0
                    and i > 0
                ):
                    raise TAEAbortException(
                        'Found additional run info "%s" in fold %d, '
                        "but cannot handle additional run info if fold >= 1."
                        % (additional_run_info, i)
                    )

                Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

                X = select(self.X_train, train_split)
                y = select(self.Y_train_targets, train_split)

                train_loss = self._loss(y, train_pred, X_data=X)
                train_losses.append(train_loss)
                # number of training data points for this fold. Used for weighting
                # the average.
                train_fold_weights.append(len(train_split))

                # Compute validation loss of this fold and store it.
                optimization_loss = self._loss(
                    self.Y_targets[i],
                    opt_pred,
                    X_data=self.X_targets[i],
                )
                opt_losses.append(optimization_loss)
                # number of optimization data points for this fold. Used for weighting
                # the average.
                opt_fold_weights.append(len(test_split))

            # Compute weights of each fold based on the number of samples in each
            # fold.
            train_fold_weights = [
                w / sum(train_fold_weights) for w in train_fold_weights
            ]
            opt_fold_weights = [w / sum(opt_fold_weights) for w in opt_fold_weights]

            train_loss = {
                metric.name: np.average(
                    [train_losses[i][str(metric)] for i in range(self.num_cv_folds)],
                    weights=train_fold_weights,
                )
                for metric in self.metrics
            }

            # if all_scoring_function is true, return a dict of opt_loss. Otherwise,
            # return a scalar.
            opt_loss = {}
            for metric_name in list(opt_losses[0].keys()) + [
                metric.name for metric in self.metrics
            ]:
                opt_loss[metric_name] = np.average(
                    [opt_losses[i][metric_name] for i in range(self.num_cv_folds)],
                    weights=opt_fold_weights,
                )

            X_targets = self.X_targets
            Y_targets = self.Y_targets
            Y_train_targets = self.Y_train_targets

            Y_optimization_pred = concat_data(
                Y_optimization_pred, num_cv_folds=self.num_cv_folds
            )
            X_targets = concat_data(X_targets, num_cv_folds=self.num_cv_folds)
            Y_targets = concat_data(Y_targets, num_cv_folds=self.num_cv_folds)

            if self.X_test is not None:
                Y_test_pred = np.array(
                    [
                        Y_test_pred[i]
                        for i in range(self.num_cv_folds)
                        if Y_test_pred[i] is not None
                    ]
                )
                # Average the predictions of several models
                if len(np.shape(Y_test_pred)) == 3:
                    Y_test_pred = np.nanmean(Y_test_pred, axis=0)

            self.X_optimization = X_targets
            self.Y_optimization = Y_targets
            self.Y_actual_train = Y_train_targets

            if self.num_cv_folds > 1:
                self.model = self._get_model(self.feat_type)
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True
                # TODO check if there might be reasons for do-not-advance here!
                status = StatusType.SUCCESS
            elif (
                self.budget_type == "iterations"
                or self.budget_type == "mixed"
                and self.model.estimator_supports_iterative_fit()
            ):
                budget_factor = self.model.get_max_iter()
                # We check for budget being None in initialization
                n_iter = int(np.ceil(cast(float, self.budget) / 100 * budget_factor))
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < n_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                if self.model.estimator_supports_iterative_fit():
                    model_max_iter = self.model.get_max_iter()
                    model_current_iter = self.model.get_current_iter()
                    if model_current_iter < model_max_iter:
                        status = StatusType.DONOTADVANCE
                    else:
                        status = StatusType.SUCCESS
                else:
                    status = StatusType.SUCCESS

            self.finish_up(
                loss=opt_loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_pred,
                test_pred=Y_test_pred if self.X_test is not None else None,
                additional_run_info=additional_run_info,
                file_output=True,
                final_call=True,
                status=status,
            )

    def partial_fit_predict_and_loss(self, fold: int, iterative: bool = False) -> None:
        """Fit, predict and get loss for eval_partial_cv (iterative and normal)"""
        if fold > self.num_cv_folds:
            raise ValueError(
                "Cannot evaluate a fold %d which is higher than "
                "the number of folds %d." % (fold, self.num_cv_folds)
            )
        if self.budget_type is not None:
            raise NotImplementedError()

        y = _get_y_array(self.Y_train, self.task_type)
        for i, (train_split, test_split) in enumerate(
            self.splitter.split(
                self.X_train, y, groups=self.resampling_strategy_args.get("groups")
            )
        ):
            if i != fold:
                continue
            else:
                break

        if self.num_cv_folds > 1:
            self.X_optimization = select(self.X_train, test_split)
            self.Y_optimization = select(self.Y_train, test_split)
            self.Y_actual_train = select(self.Y_train, train_split)

        if iterative:
            self._partial_fit_and_predict_iterative(
                fold,
                train_indices=train_split,
                test_indices=test_split,
                add_model_to_self=True,
            )
        elif self.budget_type is not None:
            raise NotImplementedError()
        else:
            (
                train_pred,
                opt_pred,
                test_pred,
                additional_run_info,
            ) = self._partial_fit_and_predict_standard(
                fold,
                train_indices=train_split,
                test_indices=test_split,
                add_model_to_self=True,
            )

            # This is my best guess at what the X_data for these should be
            X_train = select(self.X_train, train_split)  # From above (only cv?)
            X_fold = self.X_targets[fold]
            train_loss = self._loss(self.Y_actual_train, train_pred, X_data=X_train)
            loss = self._loss(self.Y_targets[fold], opt_pred, X_data=X_fold)

            if self.model.estimator_supports_iterative_fit():
                model_max_iter = self.model.get_max_iter()
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < model_max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                status = StatusType.SUCCESS

            self.finish_up(
                loss=loss,
                train_loss=train_loss,
                opt_pred=opt_pred,
                test_pred=test_pred,
                file_output=False,
                final_call=True,
                additional_run_info=None,
                status=status,
            )

    def _partial_fit_and_predict_iterative(
        self,
        fold: int,
        train_indices: List[int],
        test_indices: List[int],
        add_model_to_self: bool,
    ) -> None:
        model = self._get_model(self.feat_type)

        self.indices[fold] = (train_indices, test_indices)

        # Do only output the files in the case of iterative holdout,
        # In case of iterative partial cv, no file output is needed
        # because ensembles cannot be built
        file_output = True if self.num_cv_folds == 1 else False

        if model.estimator_supports_iterative_fit():
            X = select(self.X_train, train_indices)
            y = select(self.Y_train, train_indices)

            X_test = select(self.X_train, test_indices)
            y_test = select(self.Y_train, test_indices)

            Xt, fit_params = model.fit_transformer(X, y)

            self.Y_train_targets[train_indices] = y

            iteration = 1
            total_n_iteration = 0
            model_max_iter = model.get_max_iter()

            if self.budget is not None and self.budget > 0:
                max_n_iter_budget = int(np.ceil(self.budget / 100 * model_max_iter))
                max_iter = min(model_max_iter, max_n_iter_budget)
            else:
                max_iter = model_max_iter
            model_current_iter = 0

            while (
                not model.configuration_fully_fitted() and model_current_iter < max_iter
            ):
                n_iter = int(2**iteration / 2) if iteration > 1 else 2
                total_n_iteration += n_iter
                model.iterative_fit(Xt, y, n_iter=n_iter, **fit_params)

                (Y_train_pred, Y_optimization_pred, Y_test_pred,) = self._predict(
                    model,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )

                if add_model_to_self:
                    self.model = model

                train_loss = self._loss(y, Y_train_pred, X_data=X)
                loss = self._loss(y_test, Y_optimization_pred, X_data=X_test)

                additional_run_info = model.get_additional_run_info()

                model_current_iter = model.get_current_iter()
                if model_current_iter < max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS

                if model.configuration_fully_fitted() or model_current_iter >= max_iter:
                    final_call = True
                else:
                    final_call = False

                self.finish_up(
                    loss=loss,
                    train_loss=train_loss,
                    opt_pred=Y_optimization_pred,
                    test_pred=Y_test_pred,
                    additional_run_info=additional_run_info,
                    file_output=file_output,
                    final_call=final_call,
                    status=status,
                )
                iteration += 1

            return
        else:

            (
                Y_train_pred,
                Y_optimization_pred,
                Y_test_pred,
                additional_run_info,
            ) = self._partial_fit_and_predict_standard(
                fold, train_indices, test_indices, add_model_to_self
            )

            X = select(self.X_train, train_indices)
            y = select(self.Y_train, train_indices)
            train_loss = self._loss(y, Y_train_pred, X_data=X)

            X_test = select(self.X_train, test_indices)
            y_test = select(self.Y_train, test_indices)
            loss = self._loss(y_test, Y_optimization_pred, X_data=X_test)

            if self.model.estimator_supports_iterative_fit():
                model_max_iter = self.model.get_max_iter()
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < model_max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                status = StatusType.SUCCESS
            self.finish_up(
                loss=loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_pred,
                test_pred=Y_test_pred,
                additional_run_info=additional_run_info,
                file_output=file_output,
                final_call=True,
                status=status,
            )
            return

    def _partial_fit_and_predict_standard(
        self,
        fold: int,
        train_indices: List[int],
        test_indices: List[int],
        add_model_to_self: bool = False,
    ) -> Tuple[
        PIPELINE_DATA_DTYPE,  # train_pred
        PIPELINE_DATA_DTYPE,  # opt_pred
        PIPELINE_DATA_DTYPE,  # test_pred
        TYPE_ADDITIONAL_INFO,
    ]:
        model = self._get_model(self.feat_type)

        self.indices[fold] = (train_indices, test_indices)

        X = select(self.X_train, train_indices)
        y = select(self.Y_train, train_indices)
        _fit_and_suppress_warnings(self.logger, model, X, y)

        if add_model_to_self:
            self.model = model
        else:
            self.models[fold] = model

        self.X_targets[fold] = select(self.X_train, test_indices)
        self.Y_targets[fold] = select(self.Y_train, test_indices)
        self.Y_train_targets[train_indices] = select(self.Y_train, train_indices)

        train_pred, opt_pred, test_pred = self._predict(
            model=model,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        additional_run_info = model.get_additional_run_info()
        return (
            train_pred,
            opt_pred,
            test_pred,
            additional_run_info,
        )

    def _partial_fit_and_predict_budget(
        self,
        fold: int,
        train_indices: List[int],
        test_indices: List[int],
        add_model_to_self: bool = False,
    ) -> Tuple[
        PIPELINE_DATA_DTYPE,  # train_pred
        PIPELINE_DATA_DTYPE,  # opt_pred
        PIPELINE_DATA_DTYPE,  # test_pred
        TYPE_ADDITIONAL_INFO,
    ]:

        # This function is only called in the event budget is not None
        # Add this statement for mypy
        assert self.budget is not None

        model = self._get_model(self.feat_type)
        self.indices[fold] = (train_indices, test_indices)
        self.X_targets[fold] = select(self.X_train, test_indices)
        self.Y_targets[fold] = select(self.Y_train, test_indices)
        self.Y_train_targets[train_indices] = select(self.Y_train, train_indices)

        _fit_with_budget(
            X_train=self.X_train,
            Y_train=self.Y_train,
            budget=self.budget,
            budget_type=self.budget_type,
            logger=self.logger,
            model=model,
            train_indices=train_indices,
            task_type=self.task_type,
        )

        train_pred, opt_pred, test_pred = self._predict(
            model,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_model_to_self:
            self.model = model
        else:
            self.models[fold] = model

        additional_run_info = model.get_additional_run_info()
        return (
            train_pred,
            opt_pred,
            test_pred,
            additional_run_info,
        )

    def _predict(
        self, model: BaseEstimator, test_indices: List[int], train_indices: List[int]
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        y_train = select(self.Y_train, train_indices)
        X_train = select(self.X_train, train_indices)
        X_test = select(self.X_train, test_indices)

        # The y_train here does not correspond to the X, there to ensure output shape
        # will be correct in case labels were missing from some split
        train_pred = self.predict_function(X_train, model, self.task_type, y_train)
        opt_pred = self.predict_function(X_test, model, self.task_type, y_train)

        # This is the test data the user can pass in
        if self.X_test is not None:
            # See comment above about y_train
            X_user = self.X_test.copy()
            test_pred = self.predict_function(X_user, model, self.task_type, y_train)
        else:
            test_pred = None

        return train_pred, opt_pred, test_pred

    def get_splitter(
        self, D: AbstractDataManager
    ) -> Union[BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit]:

        if self.resampling_strategy_args is None:
            self.resampling_strategy_args = {}

        if self.resampling_strategy is not None and not isinstance(
            self.resampling_strategy, str
        ):
            if "groups" not in self.resampling_strategy_args:
                self.resampling_strategy_args["groups"] = None

            if isinstance(
                self.resampling_strategy,
                (BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit),
            ):
                self.check_splitter_resampling_strategy(
                    X=D.data["X_train"],
                    y=D.data["Y_train"],
                    groups=self.resampling_strategy_args.get("groups"),
                    task=D.info["task"],
                    resampling_strategy=self.resampling_strategy,
                )
                return self.resampling_strategy

            # If it got to this point, we are dealing with a non-supported
            # re-sampling strategy
            raise ValueError(
                "Unsupported resampling strategy {}/{} provided".format(
                    self.resampling_strategy,
                    type(self.resampling_strategy),
                )
            )

        y = D.data["Y_train"]
        shuffle = self.resampling_strategy_args.get("shuffle", True)
        train_size = 0.67
        if self.resampling_strategy_args:
            train_size_from_user = self.resampling_strategy_args.get("train_size")
            if train_size_from_user is not None:
                train_size = float(train_size_from_user)
        test_size = float("%.4f" % (1 - train_size))

        if (
            D.info["task"] in CLASSIFICATION_TASKS
            and D.info["task"] != MULTILABEL_CLASSIFICATION
        ):

            y = y.ravel()
            if self.resampling_strategy in ["holdout", "holdout-iterative-fit"]:

                if shuffle:
                    try:
                        cv = StratifiedShuffleSplit(
                            n_splits=1,
                            test_size=test_size,
                            random_state=1,
                        )
                        test_cv = copy.deepcopy(cv)
                        next(test_cv.split(y, y))
                    except ValueError as e:
                        if "The least populated class in y has only" in e.args[0]:
                            cv = CustomStratifiedShuffleSplit(
                                n_splits=1,
                                test_size=test_size,
                                random_state=1,
                            )
                        else:
                            raise e
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in [
                "cv",
                "cv-iterative-fit",
                "partial-cv",
                "partial-cv-iterative-fit",
            ]:
                if shuffle:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            cv = StratifiedKFold(
                                n_splits=self.resampling_strategy_args["folds"],
                                shuffle=shuffle,
                                random_state=1,
                            )
                            test_cv = copy.deepcopy(cv)
                            next(test_cv.split(y, y))
                    except UserWarning as e:
                        print(e)
                        if "The least populated class in y has only" in e.args[0]:
                            cv = CustomStratifiedKFold(
                                n_splits=self.resampling_strategy_args["folds"],
                                shuffle=shuffle,
                                random_state=1,
                            )
                        else:
                            raise e
                else:
                    cv = KFold(
                        n_splits=self.resampling_strategy_args["folds"], shuffle=shuffle
                    )
            else:
                raise ValueError(self.resampling_strategy)
        else:
            if self.resampling_strategy in ["holdout", "holdout-iterative-fit"]:
                # TODO shuffle not taken into account for this
                if shuffle:
                    cv = ShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in [
                "cv",
                "partial-cv",
                "partial-cv-iterative-fit",
            ]:
                random_state = 1 if shuffle else None
                cv = KFold(
                    n_splits=self.resampling_strategy_args["folds"],
                    shuffle=shuffle,
                    random_state=random_state,
                )
            else:
                raise ValueError(self.resampling_strategy)
        return cv

    @classmethod
    def check_splitter_resampling_strategy(
        cls,
        X: PIPELINE_DATA_DTYPE,
        y: np.ndarray,
        task: int,
        groups: Any,
        resampling_strategy: Union[
            BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
        ],
    ) -> None:
        if (
            task in CLASSIFICATION_TASKS
            and task != MULTILABEL_CLASSIFICATION
            or (task in REGRESSION_TASKS and task != MULTIOUTPUT_REGRESSION)
        ):
            y = y.ravel()

        try:
            resampling_strategy.get_n_splits(X=X, y=y, groups=groups)
            next(resampling_strategy.split(X=X, y=y, groups=groups))
        except Exception as e:
            raise ValueError(
                "Unsupported resampling strategy "
                "{}/{} cause exception: {}".format(
                    resampling_strategy,
                    groups,
                    str(e),
                )
            )


# create closure for evaluating an algorithm
def eval_holdout(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = 100.0,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:
    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        metrics=metrics,
        configuration=config,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        additional_components=additional_components,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )
    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_holdout(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = 100.0,
    budget_type: Optional[str] = None,
) -> None:
    return eval_holdout(
        queue=queue,
        port=port,
        config=config,
        backend=backend,
        metrics=metrics,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        instance=instance,
        disable_file_output=disable_file_output,
        iterative=True,
        additional_components=additional_components,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )


def eval_partial_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:
    if budget_type is not None:
        raise NotImplementedError()
    instance_dict: Dict[str, int] = json.loads(instance) if instance is not None else {}
    fold = instance_dict["fold"]

    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        metrics=metrics,
        configuration=config,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=False,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        additional_components=additional_components,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )

    evaluator.partial_fit_predict_and_loss(fold=fold, iterative=iterative)


def eval_partial_cv_iterative(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
) -> None:
    if budget_type is not None:
        raise NotImplementedError()
    return eval_partial_cv(
        queue=queue,
        config=config,
        backend=backend,
        metrics=metrics,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        port=port,
        num_run=num_run,
        instance=instance,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        iterative=True,
        additional_components=additional_components,
        init_params=init_params,
    )


# create closure for evaluating an algorithm
def eval_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:
    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        metrics=metrics,
        configuration=config,
        seed=seed,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        additional_components=additional_components,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )

    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[
        str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
    ],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metrics: Sequence[Scorer],
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    additional_components: Dict[str, ThirdPartyComponents],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = True,
) -> None:
    eval_cv(
        backend=backend,
        queue=queue,
        metrics=metrics,
        config=config,
        seed=seed,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        port=port,
        additional_components=additional_components,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        iterative=iterative,
        instance=instance,
    )
