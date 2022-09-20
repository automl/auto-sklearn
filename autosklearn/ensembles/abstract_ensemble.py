from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES
from autosklearn.ensemble_building.run import Run
from autosklearn.metrics import Scorer
from autosklearn.pipeline.base import BasePipeline


class AbstractEnsemble(ABC):
    @abstractmethod
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        random_state: int | np.random.RandomState | None = None,
    ):
        pass

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        return {key: value for key, value in self.__dict__.items() if key != "metrics"}

    @abstractmethod
    def fit(
        self,
        base_models_predictions: np.ndarray | List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> "AbstractEnsemble":
        """Fit an ensemble given predictions of base models and targets.

        Ensemble building maximizes performance (in contrast to
        hyperparameter optimization)!

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
            considered by the ensemble builder.

        Returns
        -------
        self

        """
        pass

    @abstractmethod
    def predict(
        self, base_models_predictions: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        base_models_predictions : np.ndarray
            shape = (n_base_models, n_data_points, n_targets)
            Same as in the fit method.

        Returns
        -------
        np.ndarray
        """
        pass

    @abstractmethod
    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        """List of (weight, model) pairs for all models included in the ensemble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        List[Tuple[float, BasePipeline]]
        """

    @abstractmethod
    def get_identifiers_with_weights(
        self,
    ) -> List[Tuple[Tuple[int, int, float], float]]:
        """Return a (identifier, weight)-pairs for all models that were passed to the
        ensemble builder.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        List[Tuple[Tuple[int, int, float], float]
        """

    @abstractmethod
    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """Return identifiers of models in the ensemble.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """

    @abstractmethod
    def get_validation_performance(self) -> float:
        """Return validation performance of ensemble.

        Returns
        -------
        float
        """


class AbstractMultiObjectiveEnsemble(AbstractEnsemble):
    @property
    @abstractmethod
    def pareto_set(self) -> Sequence[AbstractEnsemble]:
        """Get a sequence on ensembles that are on the pareto front

        Raises
        ------
        SklearnNotFittedError
            If ``fit`` has not been called and the pareto set does not exist yet

        Returns
        -------
        Sequence[AbstractEnsemble]
        """
        ...
