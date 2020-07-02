from collections import Counter
import random

import numpy as np

from autosklearn.constants import TASK_TYPES
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import calculate_score
from autosklearn.metrics import Scorer


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        task_type: int,
        metric: Scorer,
        bagging: bool = False,
        mode: str = 'fast',
        random_state: np.random.RandomState = None,
    ):
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.bagging = bagging
        self.mode = mode
        self.random_state = random_state

    def fit(self, predictions, labels, identifiers):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if self.task_type not in TASK_TYPES:
            raise ValueError('Unknown task type %s.' % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError('Metric must be of type scorer')
        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        if self.bagging:
            self._bagging(predictions, labels)
        else:
            self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(self, predictions, labels):
        if self.mode == 'fast':
            self._fast(predictions, labels)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(self, predictions, labels):
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(predictions[0].shape)
        fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction.fill(0.0)
            else:
                weighted_ensemble_prediction.fill(0.0)
                for pred in ensemble:
                    np.add(
                        weighted_ensemble_prediction,
                        pred,
                        out=weighted_ensemble_prediction,
                    )
                np.multiply(
                    weighted_ensemble_prediction,
                    1/s,
                    out=weighted_ensemble_prediction,
                )
                np.multiply(
                    weighted_ensemble_prediction,
                    (s / float(s + 1)),
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # TODO: this could potentially be vectorized! - let's profile
                # the script first!
                fant_ensemble_prediction.fill(0.0)
                np.add(
                    fant_ensemble_prediction,
                    weighted_ensemble_prediction,
                    out=fant_ensemble_prediction
                )
                np.add(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)) * pred,
                    out=fant_ensemble_prediction
                )
                scores[j] = self.metric._optimum - calculate_score(
                    solution=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]

    def _slow(self, predictions, labels):
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        for i in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = self.metric._optimum - calculate_score(
                    solution=labels,
                    prediction=ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(order)
        self.trajectory_ = np.array(trajectory)
        self.train_score_ = trajectory[-1]

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _bagging(self, predictions, labels, fraction=0.5, n_bags=20):
        """Rich Caruana's ensemble selection method with bagging."""
        raise ValueError('Bagging might not work with class-based interface!')
        n_models = predictions.shape[0]
        bag_size = int(n_models * fraction)

        order_of_each_bag = []
        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(random.sample(range(0, n_models), bag_size))
            bag = predictions[indices, :, :]
            order, _ = self._fit(bag, labels)
            order_of_each_bag.append(order)

        return np.array(order_of_each_bag)

    def predict(self, predictions):
        predictions = np.asarray(predictions)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")

    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models):
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self):
        return self.trajectory_[-1]
