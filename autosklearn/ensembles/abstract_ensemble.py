from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np

from autosklearn.pipeline.base import BasePipeline


class AbstractEnsemble(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(
        self,
        base_models_predictions: np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> 'AbstractEnsemble':
        """Fit an ensemble given predictions of base models and targets.

        Ensemble building maximizes performance (in contrast to
        hyperparameter optimization)!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        Returns
        -------
        self

        """
        pass

    @abstractmethod
    def predict(self, base_models_predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            Same as in the fit method.

        Returns
        -------
        array : [n_data_points]
        """
        self

    @abstractmethod
    def get_models_with_weights(self, models: Dict) -> List[Tuple[float, BasePipeline]]:
        """Return a list of (weight, model) pairs

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        array : [(weight_1, model_1), ..., (weight_n, model_n)]
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

        Return
        ------
        float
        """
