from __future__ import annotations

from typing import Sequence

import os

import numpy as np
from smac.runhistory.runhistory import RunHistory

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES
from autosklearn.ensemble_building.run import Run
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline


class AbstractSingleModelEnsemble(AbstractEnsemble):
    """Ensemble consisting of a single model.

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

    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.weights_ = [1.0]
        self.task_type = task_type
        if isinstance(metrics, Sequence):
            self.metrics = metrics
        elif isinstance(metrics, Scorer):
            self.metrics = [metrics]
        else:
            raise TypeError(type(metrics))
        self.random_state = random_state
        self.backend = backend

    def fit(
        self,
        base_models_predictions: np.ndarray | list[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> AbstractSingleModelEnsemble:
        """Fit the ensemble

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
            considered by the ensemble builder.

        X_data : list-like | sparse matrix | None = None

        Returns
        -------
        self
        """
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
        return "%s:\n\tMembers: %s" "\n\tWeights: %s\n\tIdentifiers: [%s]" % (
            self.__class__.__name__,
            self.indices_,  # type: ignore [attr-defined]
            self.weights_,
            self.identifiers_[0],  # type: ignore [attr-defined]
        )

    def get_models_with_weights(
        self, models: dict[tuple[int, int, float], BasePipeline]
    ) -> list[tuple[float, BasePipeline]]:
        """List of (weight, model) pairs for the model selected by this ensemble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        list[tuple[float, BasePipeline]]
        """
        return [(self.weights_[0], models[self.identifiers_[0]])]  # type: ignore [attr-defined]  # noqa: E501

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
        return list(zip(self.identifiers_, self.weights_))  # type: ignore [attr-defined]  # noqa: E501

    def get_selected_model_identifiers(self) -> list[tuple[int, int, float]]:
        """Return identifier of models in the ensemble.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        return self.identifiers_  # type: ignore [attr-defined]

    def get_validation_performance(self) -> float:
        """Return validation performance of ensemble.

        In case of multi-objective problem, only the first metric will be returned.

        Return
        ------
        float
        """
        return self.best_model_score_  # type: ignore [attr-defined]


class SingleModelEnsemble(AbstractSingleModelEnsemble):
    """Ensemble consisting of a single model.

    This class is used by the :class:`MultiObjectiveDummyEnsemble` to represent
    ensembles consisting of a single model, and this class should not be used
    on its own.

    Do not use by yourself!

    Parameters
    ----------
    task_type: int
        An identifier indicating which task is being performed.

    metrics: Sequence[Scorer] | Scorer
        The metrics used to evaluate the models.

    backend : Backend
        Gives access to the backend of Auto-sklearn. Not used.

    model_index : int
        Index of the model that constitutes the ensemble. This index will
        be used to select the correct predictions that will be passed during
        ``fit`` and ``predict``.

    random_state: int | RandomState | None = None
        Not used.
    """

    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        model_index: int,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            task_type=task_type,
            metrics=metrics,
            random_state=random_state,
            backend=backend,
        )
        self.indices_ = [model_index]

    def fit(
        self,
        base_models_predictions: np.ndarray | list[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
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

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        runs: Sequence[Run]
            Additional information for each run executed by SMAC that was
            considered by the ensemble builder. Not used.

        X_data : list-like | spmatrix | None = None
           X data to feed to a metric if it requires it

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


class SingleBest(AbstractSingleModelEnsemble):
    """Ensemble consisting of the single best model.

    Parameters
    ----------
    task_type: int
        An identifier indicating which task is being performed.

    metrics: Sequence[Scorer] | Scorer
        The metrics used to evaluate the models.

    random_state: int | RandomState | None = None
        Not used.

    backend : Backend
        Gives access to the backend of Auto-sklearn. Not used.
    """

    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            task_type=task_type,
            metrics=metrics,
            random_state=random_state,
            backend=backend,
        )

    def fit(
        self,
        base_models_predictions: np.ndarray | list[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: list[tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> SingleBest:
        """Select the single best model.

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

        X_data : array-like | sparse matrix | None = None

        Returns
        -------
         self
        """
        losses = [
            calculate_losses(
                solution=true_targets,
                prediction=base_model_prediction,
                task_type=self.task_type,
                metrics=self.metrics,
                X_data=X_data,
            )[self.metrics[0].name]
            for base_model_prediction in base_models_predictions
        ]
        argmin = np.argmin(losses)
        self.indices_ = [argmin]
        self.identifiers_ = [model_identifiers[argmin]]
        self.best_model_score_ = losses[argmin]
        return self


class SingleBestFromRunhistory(AbstractSingleModelEnsemble):
    """
    In the case of a crash, this class searches
    for the best individual model.

    Such model is returned as an ensemble of a single
    object, to comply with the expected interface of an
    AbstractEnsemble.

    Do not use by yourself!
    """

    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        run_history: RunHistory,
        seed: int,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            task_type=task_type,
            metrics=metrics,
            random_state=random_state,
            backend=backend,
        )

        # The seed here is seperate from RandomState and is used to indiicate a
        # directory for the backend to search in
        self.seed = seed
        self.indices_ = [0]
        self.weights_ = [1.0]
        self.run_history = run_history
        self.identifiers_ = self.get_identifiers_from_run_history()

    def get_identifiers_from_run_history(self) -> list[tuple[int, int, float]]:
        """Parses the run history, to identify the best performing model

        Populates the identifiers attribute, which is used by the backend to access
        the actual model.
        """
        best_model_identifier = []
        best_model_score = self.metrics[0]._worst_possible_result

        for run_key in self.run_history.data.keys():
            run_value = self.run_history.data[run_key]
            print(run_key, run_value)
            if len(self.metrics) == 1:
                cost = run_value.cost
            else:
                cost = run_value.cost[0]
            score = self.metrics[0]._optimum - (self.metrics[0]._sign * cost)

            if (score > best_model_score and self.metrics[0]._sign > 0) or (
                score < best_model_score and self.metrics[0]._sign < 0
            ):

                # Make sure that the individual best model actually exists
                model_dir = self.backend.get_numrun_directory(
                    self.seed,
                    run_value.additional_info["num_run"],
                    run_key.budget,
                )
                model_file_name = self.backend.get_model_filename(
                    self.seed,
                    run_value.additional_info["num_run"],
                    run_key.budget,
                )
                file_path = os.path.join(model_dir, model_file_name)
                if not os.path.exists(file_path):
                    continue

                best_model_identifier = [
                    (
                        self.seed,
                        run_value.additional_info["num_run"],
                        run_key.budget,
                    )
                ]
                best_model_score = score

        if not best_model_identifier:
            raise ValueError(
                "No valid model found in run history. This means smac was not able to"
                " fit a valid model. Please check the log file for errors."
            )

        self.best_model_score_ = best_model_score

        return best_model_identifier
