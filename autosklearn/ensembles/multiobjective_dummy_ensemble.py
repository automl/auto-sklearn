from __future__ import annotations

from typing import Any, Sequence

import warnings

import numpy as np

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


class MultiObjectiveDummyEnsemble(AbstractMultiObjectiveEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        backend: Backend,
    ) -> None:
        """A dummy implementation of a multi-objective ensemble.

        Builds ensembles that are individual models on the Pareto front each.

        Parameters
        ----------
        task_type: int
            An identifier indicating which task is being performed.

        metrics: Sequence[Scorer] | Scorer
            The metrics used to evaluate the models.

        random_state: Optional[int | RandomState] = None
            Not used.

        backend : Backend
            Gives access to the backend of Auto-sklearn. Not used.
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

    def __getstate__(self) -> dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        return {key: value for key, value in self.__dict__.items() if key != "metrics"}

    def fit(
        self,
        base_models_predictions: list[np.ndarray],
        X_data: SUPPORTED_FEAT_TYPES | None,
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
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

        X_data : list-like or sparse data

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        runs: Sequence[Run]
            Additional information for each run executed by SMAC that was
            considered by the ensemble builder. Not used.

        Returns
        -------
        self
        """
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)

        def is_pareto_efficient_simple(costs: np.ndarray) -> np.ndarray:
            """
            Plot the Pareto Front in our 2d example.

            source from: https://stackoverflow.com/a/40239615
            Find the pareto-efficient points

            Parameters
            ----------
            costs: np.ndarray

            Returns
            -------
            np.ndarray
            """

            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    # Keep any point with a lower cost
                    is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)

                    # And keep self
                    is_efficient[i] = True
            return is_efficient

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
        efficient_points = is_pareto_efficient_simple(all_costs)
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
        return self.pareto_set_[0].predict(base_models_predictions)

    def __str__(self) -> str:
        return "MultiObjectiveDummyEnsemble: %d models" % len(self.pareto_set_)

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
        return self.pareto_set_[0].get_models_with_weights(models)

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
        return self.pareto_set_[0].get_identifiers_with_weights()

    def get_selected_model_identifiers(self) -> list[tuple[int, int, float]]:
        """Return identifiers of models in the ensemble that is best for the 1st metric.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        return self.pareto_set_[0].get_selected_model_identifiers()

    def get_validation_performance(self) -> float:
        """Return validation performance of the ensemble that is best for the 1st metric.

        Returns
        -------
        float
        """
        return self.pareto_set_[0].get_validation_performance()

    def get_pareto_set(self) -> Sequence[AbstractEnsemble]:
        return self.pareto_set_
