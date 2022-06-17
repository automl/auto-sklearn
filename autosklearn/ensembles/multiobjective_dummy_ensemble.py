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
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline


class SingleModelEnsemble(AbstractEnsemble):
    """Ensemble consisting of a single model.

    This class is used my the :cls:`MultiObjectiveDummyEnsemble` to represent ensembles
    consisting of a single model, and this class should not be used on its own.

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

    model_index : int
        Index of the model that constitutes the ensemble. This index will
        be used to select the correct predictions that will be passed during
        ``fit`` and ``predict``.
    """

    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        backend: Backend,
        model_index: int,
    ):
        # Add some default values -- at least 1 model in ensemble is assumed
        self.indices_ = [model_index]
        self.weights_ = [1.0]
        self.task_type = task_type
        self.metrics = metrics if isinstance(metrics, Sequence) else [metrics]

    def fit(
        self,
        base_models_predictions: np.ndarray | list[np.ndarray],
        X_data: SUPPORTED_FEAT_TYPES | None,
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
    ) -> SingleModelEnsemble:
        """Dummy implementation of the ``fit`` method.

        Actualy work of passing the model index is done in the constructor. This
        method only stores the identifier of the selected model and computes it's
        validation loss.

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
        self.identifiers_ = [model_identifiers[self.indices_[0]]]
        loss = calculate_losses(
            solution=true_targets,
            prediction=base_models_predictions[self.indices_[0]],
            task_type=self.task_type,
            metrics=self.metrics,
            X_data=X_data,
        )
        self.best_model_score_ = loss[self.metrics[0].name]
        return self

    def predict(self, predictions: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Select the predictions of the selected model.

        Parameters
        ----------
        base_models_predictions : np.ndarray
            shape = (n_base_models, n_data_points, n_targets)
            Same as in the fit method.

        Returns
        -------
        np.ndarray
        """
        return predictions[0]

    def __str__(self) -> str:
        return "Single Model:\n\tMembers: %s" "\n\tWeights: %s\n\tIdentifiers: %s" % (
            self.indices_,
            self.weights_,
            " ".join(
                [
                    str(identifier)
                    for idx, identifier in enumerate(self.identifiers_)
                    if self.weights_[idx] > 0
                ]
            ),
        )

    def get_models_with_weights(
        self, models: dict[tuple[int, int, float], BasePipeline]
    ) -> list[tuple[float, BasePipeline]]:
        """Return a list of (weight, model) pairs for the model selected by this ensemble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        list[tuple[float, BasePipeline]]
        """
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_identifiers_with_weights(
        self,
    ) -> list[tuple[tuple[int, int, float], float]]:
        """Return a (identifier, weight)-pairs for the model selected by this ensemble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        list[tuple[tuple[int, int, float], float]
        """
        return list(zip(self.identifiers_, self.weights_))

    def get_selected_model_identifiers(self) -> list[tuple[int, int, float]]:
        """Return identifier of models in the ensemble.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        """Return validation performance of ensemble.

        In case of multi-objective problem, only the first metric will be returned.

        Return
        ------
        float
        """
        return self.best_model_score_


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
        return "MultiObjectiveDummyEnsemble:\n" + (
            "\n".join([str(ensemble) for ensemble in self.pareto_set_])
        )

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
        output = []
        for i, weight in enumerate(self.pareto_set_[0].weights_):
            if weight > 0.0:
                identifier = self.pareto_set_[0].identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

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
        return list(zip(self.pareto_set_[0].identifiers_, self.pareto_set_[0].weights_))

    def get_selected_model_identifiers(self) -> list[tuple[int, int, float]]:
        """Return identifiers of models in the ensemble that is best for the 1st metric.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        output = []

        for i, weight in enumerate(self.pareto_set_[0].weights_):
            identifier = self.pareto_set_[0].identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        """Return validation performance of the ensemble that is best for the 1st metric.

        Return
        ------
        float
        """
        return self.pareto_set_[0].get_validation_performance()

    def get_pareto_set(self) -> Sequence[AbstractEnsemble]:
        return self.pareto_set_
