from collections import Counter
import random

import numpy as np
import six

from autosklearn.constants import *
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.evaluation.util import calculate_score


class EnsembleSelection(AbstractEnsemble):
    def __init__(self, ensemble_size, task_type, metric,
                 sorted_initialization=False, bagging=False, mode='fast'):
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.mode = mode

    def fit(self, predictions, labels, identifiers):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not self.task_type in TASK_TYPES:
            raise ValueError('Unknown task type %s.' % self.task_type)
        if not self.metric in METRIC:
            raise ValueError('Unknown metric %s.' % self.metric)
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

        if self.sorted_initialization:
            n_best = 20
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = calculate_score(
                    labels, ensemble_, self.task_type, self.metric,
                    ensemble_.shape[1])
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            for j, pred in enumerate(predictions):
                fant_ensemble_prediction = weighted_ensemble_prediction + \
                                           (1. / float(s + 1)) * pred
                scores[j] = calculate_score(
                    labels, fant_ensemble_prediction, self.task_type,
                    self.metric, fant_ensemble_prediction.shape[1])
            best = np.nanargmax(scores)
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

        if self.sorted_initialization:
            n_best = 20
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = calculate_score(
                    labels, ensemble_, self.task_type, self.metric,
                    ensemble_.shape[1])
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = calculate_score(labels, ensemble_prediction,
                                            self.task_type, self.metric,
                                            ensemble_prediction.shape[1])
                ensemble.pop()
            best = np.nanargmax(scores)
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

    def _sorted_initialization(self, predictions, labels, n_best):
        perf = np.zeros([predictions.shape[0]])

        for idx, prediction in enumerate(predictions):
            perf[idx] = calculate_score(labels, prediction, self.task_type,
                                        self.metric, predictions.shape[1])

        indices = np.argsort(perf)[perf.shape[0] - n_best:]
        return indices

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
        for i, weight in enumerate(self.weights_):
            predictions[i] *= weight
        return np.sum(predictions, axis=0)

    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def pprint_ensemble_string(self, models):
        output = []
        sio = six.StringIO()
        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            model = models[identifier]
            if weight > 0.0:
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        sio.write("[")
        for weight, model in output:
            sio.write("(%f, %s),\n" % (weight, model))
        sio.write("]")

        return sio.getvalue()

    def get_model_identifiers(self):
        return self.identifiers_
