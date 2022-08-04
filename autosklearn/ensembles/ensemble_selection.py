from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import random
import warnings
from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import TASK_TYPES
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES
from autosklearn.ensemble_building.run import Run
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        ensemble_size: int = 50,
        bagging: bool = False,
        mode: str = "fast",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        """An ensemble of selected algorithms

        Fitting an EnsembleSelection generates an ensemble from the the models
        generated during the search process. Can be further used for prediction.

        Parameters
        ----------
        task_type: int
            An identifier indicating which task is being performed.

        metrics: Sequence[Scorer] | Scorer
            The metric used to evaluate the models. If multiple metrics are passed,
            ensemble selection only optimizes for the first

        backend : Backend
            Gives access to the backend of Auto-sklearn. Not used by Ensemble Selection.

        bagging: bool = False
            Whether to use bagging in ensemble selection

        mode: str in ['fast', 'slow'] = 'fast'
            Which kind of ensemble generation to use
            * 'slow' - The original method used in Rich Caruana's ensemble selection.
            * 'fast' - A faster version of Rich Caruanas' ensemble selection.

        random_state: int | RandomState | None = None
            The random_state used for ensemble selection.

            * None - Uses numpy's default RandomState object
            * int - Successive calls to fit will produce the same results
            * RandomState - Truly random, each call to fit will produce
              different results, even with the same object.

        References
        ----------
        | Ensemble selection from libraries of models
        | Rich Caruana, Alexandru Niculescu-Mizil, Geoff Crew and Alex Ksikes
        | ICML 2004
        | https://dl.acm.org/doi/10.1145/1015330.1015432
        | https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
        """  # noqa: E501
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        if isinstance(metrics, Sequence):
            if len(metrics) > 1:
                warnings.warn(
                    "Ensemble selection can only optimize one metric, "
                    "but multiple metrics were passed, dropping all "
                    "except for the first metric."
                )
            self.metric = metrics[0]
        else:
            self.metric = metrics
        self.bagging = bagging
        self.mode = mode

        # Behaviour similar to sklearn
        #   int - Deteriministic with succesive calls to fit
        #   RandomState - Successive calls to fit will produce differences
        #   None - Uses numpmys global singleton RandomState
        # https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
        self.random_state = random_state

    def fit(
        self,
        base_models_predictions: List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> EnsembleSelection:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError(
                "The provided metric must be an instance of Scorer, "
                "nevertheless it is {}({})".format(
                    self.metric,
                    type(self.metric),
                )
            )
        if self.mode not in ("fast", "slow"):
            raise ValueError("Unknown mode %s" % self.mode)

        if self.bagging:
            self._bagging(base_models_predictions, true_targets)
        else:
            self._fit(
                predictions=base_models_predictions,
                X_data=X_data,
                labels=true_targets,
            )
        self._calculate_weights()
        self.identifiers_ = model_identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> EnsembleSelection:
        if self.mode == "fast":
            self._fast(predictions=predictions, X_data=X_data, labels=labels)
        else:
            self._slow(predictions=predictions, X_data=X_data, labels=labels)
        return self

    def _fast(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            losses = np.zeros(
                (len(predictions)),
                dtype=np.float64,
            )
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be
                #
                #   ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                #
                # We overwrite the contents of fant_ensemble_prediction directly with
                # weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                np.multiply(
                    fant_ensemble_prediction,
                    (1.0 / float(s + 1)),
                    out=fant_ensemble_prediction,
                )

                losses[j] = calculate_losses(
                    solution=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metrics=[self.metric],
                    X_data=X_data,
                    scoring_functions=None,
                )[self.metric.name]

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()

            best = rand.choice(all_best)

            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

    def _slow(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> None:
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        for i in range(ensemble_size):
            losses = np.zeros(
                [np.shape(predictions)[0]],
                dtype=np.float64,
            )
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                losses[j] = calculate_losses(
                    solution=labels,
                    prediction=ensemble_prediction,
                    task_type=self.task_type,
                    metrics=[self.metric],
                    X_data=X_data,
                    scoring_functions=None,
                )[self.metric.name]
                ensemble.pop()
            best = np.nanargmin(losses)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(
            order,
            dtype=np.int64,
        )
        self.trajectory_ = np.array(
            trajectory,
            dtype=np.float64,
        )
        self.train_loss_ = trajectory[-1]

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _bagging(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        fraction: float = 0.5,
        n_bags: int = 20,
    ) -> np.ndarray:
        """Rich Caruana's ensemble selection method with bagging."""
        raise ValueError("Bagging might not work with class-based interface!")
        n_models = predictions.shape[0]
        bag_size = int(n_models * fraction)

        order_of_each_bag = []
        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(random.sample(range(0, n_models), bag_size))
            bag = predictions[indices, :, :]
            order, _ = self._fit(predictions=bag, labels=labels)
            order_of_each_bag.append(order)

        return np.array(
            order_of_each_bag,
            dtype=np.int64,
        )

    def predict(
        self, base_models_predictions: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:

        average = np.zeros_like(base_models_predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(base_models_predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(base_models_predictions) == len(self.weights_):
            for pred, weight in zip(base_models_predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(base_models_predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(base_models_predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError(
                "The dimensions of ensemble predictions"
                " and ensemble weights do not match!"
            )
        del tmp_predictions
        return average

    def __str__(self) -> str:
        trajectory_str = " ".join(
            [f"{id}: {perf:.5f}" for id, perf in enumerate(self.trajectory_)]
        )
        identifiers_str = " ".join(
            [
                f"{identifier}"
                for idx, identifier in enumerate(self.identifiers_)
                if self.weights_[idx] > 0
            ]
        )
        return (
            "Ensemble Selection:\n"
            f"\tTrajectory: {trajectory_str}\n"
            f"\tMembers: {self.indices_}\n"
            f"\tWeights: {self.weights_}\n"
            f"\tIdentifiers: {identifiers_str}\n"
        )

    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
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
    ) -> List[Tuple[Tuple[int, int, float], float]]:
        return list(zip(self.identifiers_, self.weights_))

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
