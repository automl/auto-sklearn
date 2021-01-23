import numpy as np

import pytest

from autosklearn.constants import BINARY_CLASSIFICATION, REGRESSION
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import accuracy, root_mean_squared_error


def testEnsembleSelection():
    """
    Makes sure ensemble selection fit method creates an ensemble correctly
    """

    ensemble = EnsembleSelection(ensemble_size=10,
                                 task_type=REGRESSION,
                                 random_state=np.random.RandomState(0),
                                 metric=root_mean_squared_error)

    # We create a problem such that we encourage the addition of members to the ensemble
    # Fundamentally, the average of 10 sequential number is 5.5
    y_true = np.full((100), 5.5)
    predictions = []
    for i in range(1, 20):
        pred = np.full((100), i, dtype=np.float32)
        pred[i*5:5*(i+1)] = 5.5 * i
        predictions.append(pred)

    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(20)])

    np.testing.assert_array_equal(ensemble.weights_,
                                  np.array([0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                           0.,  0., 0.]))

    assert ensemble.identifiers_ == [(i, i, i) for i in range(20)]

    np.testing.assert_array_almost_equal(np.array(ensemble.trajectory_),
                                         np.array([3.462296925452813, 2.679202306657711,
                                                  2.2748626436960375, 2.065717187806695,
                                                  1.7874562615598728, 1.6983448128441783,
                                                  1.559451106330085, 1.5316326052614575,
                                                  1.3801950121782542, 1.3554980575295374]))


def testPredict():
    # Test that ensemble prediction applies weights correctly to given
    # predictions. There are two possible cases:
    # 1) predictions.shape[0] == len(self.weights_). In this case,
    # predictions include those made by zero-weighted models. Therefore,
    # we simply apply each weights to the corresponding model preds.
    # 2) predictions.shape[0] < len(self.weights_). In this case,
    # predictions exclude those made by zero-weighted models. Therefore,
    # we first exclude all occurrences of zero in self.weights_, and then
    # apply the weights.
    # If none of the above is the case, predict() raises Error.
    ensemble = EnsembleSelection(ensemble_size=3,
                                 task_type=BINARY_CLASSIFICATION,
                                 random_state=np.random.RandomState(0),
                                 metric=accuracy,
                                 )
    # Test for case 1. Create (3, 2, 2) predictions.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # Weights of 3 hypothetical models
    ensemble.weights_ = [0.7, 0.2, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array([[0.89, 0.11],  # This should be the true prediction.
                      [0.35, 0.65]])
    assert np.allclose(pred, truth)

    # Test for case 2.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # The third model now has weight of zero.
    ensemble.weights_ = [0.7, 0.2, 0.0, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array([[0.89, 0.11],
                      [0.35, 0.65]])
    assert np.allclose(pred, truth)

    # Test for error case.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # Now the weights have 2 zero weights and 2 non-zero weights,
    # which is incompatible.
    ensemble.weights_ = [0.6, 0.0, 0.0, 0.4]

    with pytest.raises(ValueError):
        ensemble.predict(per_model_pred)
