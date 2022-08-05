from __future__ import annotations

from typing import Sequence

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import TASK_TYPES
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES
from autosklearn.ensemble_building.run import Run
from autosklearn.ensembles.abstract_ensemble import (
    AbstractEnsemble,
    AbstractMultiObjectiveEnsemble,
)
from autosklearn.ensembles.singlebest_ensemble import SingleModelEnsemble
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline
from autosklearn.util.multiobjective import pareto_front


class MultiObjectiveDummyEnsemble(AbstractMultiObjectiveEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        """A dummy implementation of a multi-objective ensemble.

        Builds ensembles that are individual models on the Pareto front each.

        Parameters
        ----------
        task_type: int
            An identifier indicating which task is being performed.

        metrics: Sequence[Scorer] | Scorer
            The metrics used to evaluate the models.

        backend : Backend
            Gives access to the backend of Auto-sklearn. Not used.

        random_state: int | RandomState | None = None
            Not used.
        """
        self.task_type = task_type
        if isinstance(metrics, Sequence):
            if len(metrics) == 1:
                warnings.warn(
                    "Passed only a single metric to a multi-objective ensemble. "
                    "Please use a single-objective ensemble in such cases."
                )
            self.metrics = metrics
        else:
            self.metric = [metrics]
        self.random_state = random_state
        self.backend = backend

    @property
    def pareto_set(self) -> Sequence[AbstractEnsemble]:
        if not hasattr(self, "pareto_set_"):
            raise NotFittedError("`pareto_set` not created, please call `fit()` first")
        return self.pareto_set_

    def fit(
        self,
        base_models_predictions: list[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> MultiObjectiveDummyEnsemble:
        """Select dummy ensembles given predictions of base models and targets.

        Parameters
        ----------
        base_models_predictions: np.ndarray
            shape = (n_base_models, n_data_points, n_targets)
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

            Can be a list of 2d numpy arrays as well to prevent copying all
            predictions into a single, large numpy array.

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        runs: Sequence[Run]
            Additional information for each run executed by SMAC that was
            considered by the ensemble builder. Not used.

        X_data : list-like | sparse matrix | None = None
            X data to give to the metric if required

        Returns
        -------
        self
        """
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)

        all_costs = np.empty((len(base_models_predictions), len(self.metrics)))
        for i, base_model_prediction in enumerate(base_models_predictions):
            losses = calculate_losses(
                solution=true_targets,
                prediction=base_model_prediction,
                task_type=self.task_type,
                metrics=self.metrics,
                X_data=X_data,
            )
            all_costs[i] = [losses[metric.name] for metric in self.metrics]

        all_costs = np.array(all_costs)
        sort_by_first_metric = np.argsort(all_costs[:, 0])
        efficient_points = pareto_front(all_costs, is_loss=True)
        pareto_set = []

        for argsort_idx in sort_by_first_metric:
            if not efficient_points[argsort_idx]:
                continue
            ensemble = SingleModelEnsemble(
                task_type=self.task_type,
                metrics=self.metrics,
                random_state=self.random_state,
                backend=self.backend,
                model_index=argsort_idx,
            )
            ensemble.fit(
                base_models_predictions=base_models_predictions,
                true_targets=true_targets,
                model_identifiers=model_identifiers,
                runs=runs,
                X_data=X_data,
            )
            pareto_set.append(ensemble)

        self.pareto_set_ = pareto_set
        return self

    def predict(
        self, base_models_predictions: np.ndarray | list[np.ndarray]
    ) -> np.ndarray:
        """Predict using the ensemble which is best for the 1st metric.

        Parameters
        ----------
        base_models_predictions : np.ndarray
            shape = (n_base_models, n_data_points, n_targets)
            Same as in the fit method.

        Returns
        -------
        np.ndarray
        """
        return self.pareto_set[0].predict(base_models_predictions)

    def __str__(self) -> str:
        return "MultiObjectiveDummyEnsemble: %d models" % len(self.pareto_set)

    def get_models_with_weights(
        self, models: dict[tuple[int, int, float], BasePipeline]
    ) -> list[tuple[float, BasePipeline]]:
        """Return a list of (weight, model) pairs for the ensemble that is
        best for the 1st metric.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        list[tuple[float, BasePipeline]]
        """
        return self.pareto_set[0].get_models_with_weights(models)

    def get_identifiers_with_weights(
        self,
    ) -> list[tuple[tuple[int, int, float], float]]:
        """Return a (identifier, weight)-pairs for all models that were passed to the
        ensemble builder based on the ensemble that is best for the 1st metric.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        list[tuple[tuple[int, int, float], float]
        """
        return self.pareto_set[0].get_identifiers_with_weights()

    def get_selected_model_identifiers(self) -> list[tuple[int, int, float]]:
        """Return identifiers of models in the ensemble that is best for the 1st metric.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        return self.pareto_set[0].get_selected_model_identifiers()

    def get_validation_performance(self) -> float:
        """Validation performance of the ensemble that is best for the 1st metric.

        Returns
        -------
        float
        """
        return self.pareto_set[0].get_validation_performance()
