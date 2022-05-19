from typing import Any, Dict, List, Optional, Tuple, Union

import random
from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from autosklearn.constants import TASK_TYPES
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        task_type: int,
        metric: Scorer,
        bagging: bool = False,
        mode: str = "fast",
        use_best: bool = False,
        tie_breaker_default: str = "random",
        tie_breaker_metric: Optional[Scorer] = None,
        round_losses: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """An ensemble of selected algorithms

        Fitting an EnsembleSelection generates an ensemble from the the models
        generated during the search process. Can be further used for prediction.

        Parameters
        ----------
        task_type: int
            An identifier indicating which task is being performed.
        metric: Scorer
            The metric used to evaluate the selected ensembles. Used as a loss.
        bagging: bool = False
            Whether to use bagging in ensemble selection
        mode: str in ['fast', 'slow'] = 'fast'
            Which kind of ensemble generation to use
            *   'slow' - The original method used in Rich Caruana's ensemble selection.
            *   'fast' - A faster version of Rich Caruanas' ensemble selection.
        use_best: bool = False
            If True, ensemble selection returns the best ensemble found during the
            greedy search. If False, it returns the last ensemble found. This dynamical
            changes the ensemble_size to a number in [1, ensemble_size].
        tie_breaker_default: str in ['random', 'first']
            The default tie breaker strategy that is used.
            *   'random' - Randomly select an element from the list of best ensembles in
                           case of ties. This can generate better weight vectors (better
                            ensemble performance).
            *   'first' - Select the first element from the list of best ensembles in
                          case of ties. This can generate smaller ensembles such that
                          less base models are needed (better ensemble efficiency).
            The default tie breaker strategy is also used if a second metric can not
            break the tie fully. To some extent, the selected strategy is always the
             fall-back or final tie breaker.
        tie_breaker_metric: Scorer = None
            If None, ensemble ties are broken mainly based on tie_breaker_default.
            If not None, a Scorer metric is expected. This metric is used to break ties
            of ensembles. If any ties remain, tie_breaker_default is used.
        round_losses: bool = False
            If True, the loss computed during the evaluation of an ensemble is rounded
            to 6 decimals. Rounding is disabled if the loss values are very small
            (<1e-4) to avoid falsifying the result for evaluations with small losses.

            Rounding avoids that the loss will differ only as a result of floating-point
            errors. Assuming multiple losses are equal (with perfect precision) but were
            computed with different starting values by different base models, it might
            be that a floating-point error makes one loss smaller than the other. Then,
            the minimum would be selected and instead of selecting all models with equal
            loss, only the one model with the smallest loss is selected because it had
            more "luck" with floating-point errors.

            This is bad because it avoids ties, which we could break based on a second
            metric or which we could break by randomness/first selection. In the case
            of a second metric, the scores might differ more drastically and thus we
            want to round the losses to achieve the tie. In the case of randomness, the
            probability to select a base model would be incorrect without rounding (if a
            model is not in the set of ties even with theoretical equal loss, it has no
            chance of being selected). In case of selecting the first element from a
            tie, the ensemble size would be unnecessarily inflated without rounding.


        random_state: Optional[int | RandomState] = None
            The random_state used for ensemble selection.
            *   None - Uses numpy's default RandomState object
            *   int - Successive calls to fit will produce the same results
            *   RandomState - Truely random, each call to fit will produce
                              different results, even with the same object.
        """
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.bagging = bagging
        self.mode = mode
        self.use_best = use_best
        self.tie_breaker_default = tie_breaker_default
        self.tie_breaker_metric = tie_breaker_metric
        self.round_losses = round_losses

        # Behaviour similar to sklearn
        #   int - Deteriministic with succesive calls to fit
        #   RandomState - Successive calls to fit will produce differences
        #   None - Uses numpmys global singleton RandomState
        # https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
        self.random_state = random_state

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.metric = None  # type: ignore
        self.tie_breaker_metric = None  # type: ignore
        return self.__dict__

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
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
        if self.tie_breaker_default not in ("random", "first"):
            raise ValueError(
                "Unknown tie_breaker_default %s" % self.tie_breaker_default
            )
        if (self.tie_breaker_metric is not None) and (
            not isinstance(self.tie_breaker_metric, Scorer)
        ):
            raise ValueError(
                "The provided tie_breaker_metric must be an instance of Scorer, "
                "nevertheless it is {}({})".format(
                    self.tie_breaker_metric,
                    type(self.tie_breaker_metric),
                )
            )

        if self.bagging:
            self._bagging(predictions, labels)
        else:
            self._fit(predictions, labels)

        if self.use_best:
            self._select_best_ensemble()
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> AbstractEnsemble:
        if self.mode == "fast":
            self._fast(predictions, labels)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []
        rounding_epsilon = 1e-4
        round_to_decimals = 6
        tie_break_random = self.tie_breaker_default == "random"

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
                    scoring_functions=None,
                )[self.metric.name]

            # Rounding Losses
            if self.round_losses and (np.abs(np.nanmin(losses)) > rounding_epsilon):
                losses = losses.round(round_to_decimals)

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()

            # Tie breaking
            if len(all_best) > 1:

                # Break Ties with a second metric
                if self.tie_breaker_metric is not None:
                    losses_tiebreak = np.zeros((len(all_best)), dtype=np.float64)
                    fant_ensemble_prediction_tiebreak = np.zeros(
                        weighted_ensemble_prediction.shape, dtype=np.float64
                    )
                    predictions_tiebreak = [
                        predictions[tied_pred_idx] for tied_pred_idx in all_best
                    ]

                    # Default Eval Loop from above
                    # TODO do we want this to become a function?
                    for j, pred in enumerate(predictions_tiebreak):
                        np.add(
                            weighted_ensemble_prediction,
                            pred,
                            out=fant_ensemble_prediction_tiebreak,
                        )
                        np.multiply(
                            fant_ensemble_prediction_tiebreak,
                            (1.0 / float(s + 1)),
                            out=fant_ensemble_prediction_tiebreak,
                        )

                        losses_tiebreak[j] = calculate_losses(
                            solution=labels,
                            prediction=fant_ensemble_prediction_tiebreak,
                            task_type=self.task_type,
                            metrics=[self.tie_breaker_metric],
                            scoring_functions=None,
                        )[self.tie_breaker_metric.name]

                    all_best_tied = np.argwhere(
                        losses_tiebreak == np.nanmin(losses_tiebreak)
                    ).flatten()
                    all_best = all_best[all_best_tied]

            # Select one element of all_best
            # The default case for tie-breaking: select random; else first element
            best = rand.choice(all_best) if tie_break_random else all_best[0]

            # Save selection data
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

    def _slow(self, predictions: List[np.ndarray], labels: np.ndarray) -> None:
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

    def _select_best_ensemble(self):
        """Select the best ensemble based on the trajectories"""

        # Get best ensemble (assuming trajectories are losses as above)
        idx_best_ensemble = self.trajectory_.index(np.min(self.trajectory_))

        # Make the object only keep all data up until the best ensemble
        self.indices_ = self.indices_[: idx_best_ensemble + 1]
        self.ensemble_size = idx_best_ensemble + 1
        self.train_loss_ = self.trajectory_[idx_best_ensemble]
        self.trajectory_ = self.trajectory_[: idx_best_ensemble + 1]

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
            order, _ = self._fit(bag, labels)
            order_of_each_bag.append(order)

        return np.array(
            order_of_each_bag,
            dtype=np.int64,
        )

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
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
        self, models: BasePipeline
    ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
