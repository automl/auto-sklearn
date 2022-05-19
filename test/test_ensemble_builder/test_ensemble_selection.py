import numpy as np

from autosklearn.constants import BINARY_CLASSIFICATION, REGRESSION
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import accuracy, balanced_accuracy, root_mean_squared_error

import pytest


def testEnsembleSelection():
    """
    Makes sure ensemble selection fit method creates an ensemble correctly
    """

    ensemble = EnsembleSelection(
        ensemble_size=10,
        task_type=REGRESSION,
        random_state=0,
        metric=root_mean_squared_error,
        tie_breaker_default="random",
        use_best=False,
        tie_breaker_metric=None,
        round_losses=False,
    )

    # We create a problem such that we encourage the addition of members to the ensemble
    # Fundamentally, the average of 10 sequential number is 5.5
    y_true = np.full((100), 5.5)
    predictions = []
    for i in range(1, 20):
        pred = np.full((100), i, dtype=np.float32)
        pred[i * 5 : 5 * (i + 1)] = 5.5 * i
        predictions.append(pred)

    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(1, 20)])

    np.testing.assert_array_equal(
        ensemble.weights_,
        np.array(
            [
                0.1,
                0.2,
                0.2,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert ensemble.identifiers_ == [(i, i, i) for i in range(1, 20)]

    np.testing.assert_array_almost_equal(
        np.array(ensemble.trajectory_),
        np.array(
            [
                3.462296925452813,
                2.679202306657711,
                2.2748626436960375,
                2.065717187806695,
                1.7874562615598728,
                1.6983448128441783,
                1.559451106330085,
                1.5316326052614575,
                1.3801950121782542,
                1.3554980575295374,
            ]
        ),
    )


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
    ensemble = EnsembleSelection(
        ensemble_size=3,
        task_type=BINARY_CLASSIFICATION,
        random_state=0,
        metric=accuracy,
        tie_breaker_default="random",
        use_best=False,
        tie_breaker_metric=None,
        round_losses=False,
    )
    # Test for case 1. Create (3, 2, 2) predictions.
    per_model_pred = np.array(
        [[[0.9, 0.1], [0.4, 0.6]], [[0.8, 0.2], [0.3, 0.7]], [[1.0, 0.0], [0.1, 0.9]]]
    )
    # Weights of 3 hypothetical models
    ensemble.weights_ = [0.7, 0.2, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array(
        [[0.89, 0.11], [0.35, 0.65]]  # This should be the true prediction.
    )
    assert np.allclose(pred, truth)

    # Test for case 2.
    per_model_pred = np.array(
        [[[0.9, 0.1], [0.4, 0.6]], [[0.8, 0.2], [0.3, 0.7]], [[1.0, 0.0], [0.1, 0.9]]]
    )
    # The third model now has weight of zero.
    ensemble.weights_ = [0.7, 0.2, 0.0, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array([[0.89, 0.11], [0.35, 0.65]])
    assert np.allclose(pred, truth)

    # Test for error case.
    per_model_pred = np.array(
        [[[0.9, 0.1], [0.4, 0.6]], [[0.8, 0.2], [0.3, 0.7]], [[1.0, 0.0], [0.1, 0.9]]]
    )
    # Now the weights have 2 zero weights and 2 non-zero weights,
    # which is incompatible.
    ensemble.weights_ = [0.6, 0.0, 0.0, 0.4]

    with pytest.raises(ValueError):
        ensemble.predict(per_model_pred)


def testEnsembleSelectionUseBest():
    """
    Makes sure ensemble selection use_best works as intended
    """

    ensemble = EnsembleSelection(
        ensemble_size=2,
        task_type=REGRESSION,
        random_state=0,
        metric=root_mean_squared_error,
        tie_breaker_default="random",
        use_best=False,
        round_losses=False,
        tie_breaker_metric=None,
    )
    ensemble_use_best = EnsembleSelection(
        ensemble_size=10,
        task_type=REGRESSION,
        random_state=0,
        metric=root_mean_squared_error,
        tie_breaker_default="random",
        use_best=True,
        round_losses=False,
        tie_breaker_metric=None,
    )

    # Problem where we know when to stop: with [5,6] (indices: 4,5)
    # That is the first occurrence of zero error, every 2nd time afterwards is zero too.
    # We want to select the first occurrence because a smaller ensemble is better.
    y_true = np.full((100), 5.5)
    predictions = []
    for i in range(1, 20):
        pred = np.full((100), i, dtype=np.float32)
        predictions.append(pred)

    # Call fit
    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(1, 20)])
    ensemble_use_best.fit(
        predictions, y_true, identifiers=[(i, i, i) for i in range(1, 20)]
    )

    # Check values of ensemble_use_best
    assert ensemble_use_best.ensemble_size == 2
    assert ensemble_use_best.indices_ == [4, 5]
    assert ensemble_use_best.train_loss_ == 0
    np.testing.assert_array_almost_equal(
        ensemble_use_best.trajectory_, np.array([0.5, 0])
    )
    np.testing.assert_array_equal(
        ensemble_use_best.weights_,
        np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    # Check equality of object by hand (to guarantee that use best returns a model that
    #   looks like it has run for less iterations than it actually did)
    assert ensemble.ensemble_size == ensemble_use_best.ensemble_size
    assert ensemble.indices_ == ensemble_use_best.indices_
    assert ensemble.train_loss_ == ensemble_use_best.train_loss_
    np.testing.assert_array_almost_equal(
        ensemble.trajectory_, ensemble_use_best.trajectory_
    )
    np.testing.assert_array_equal(ensemble.weights_, ensemble_use_best.weights_)

    # Check if use best did not break anything with regards to the identifiers
    assert ensemble.identifiers_ == [(i, i, i) for i in range(1, 20)]
    assert ensemble_use_best.identifiers_ == [(i, i, i) for i in range(1, 20)]


def testEnsembleSelectionTieBreaker():
    """
    Makes sure ensemble selection tie breaking works as intended
    """

    ensemble = EnsembleSelection(
        ensemble_size=10,
        task_type=BINARY_CLASSIFICATION,
        random_state=0,
        metric=accuracy,
        tie_breaker_default="random",
        use_best=False,
        tie_breaker_metric=balanced_accuracy,
        round_losses=False,
    )

    # Classification scenario where a tie for accuracy happens at the first selection.
    # This tie will be broken by the second metric (balanced accuracy).
    y_true = np.full((100), 1)
    y_true[:40] = 0

    predictions = [np.full((100), 1), np.full((100), 1)]
    predictions[1][20:60] = 0

    # Call fit
    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(2)])

    # If tie_breaker_metric would be None, in the first iteration predictions[0] would
    #   be 0, because argmin select the first index with a minimal value.
    # Hence, we can test if the tie was broken as intended with the following assertion
    #   while testing that everything worked as intended.
    assert ensemble.indices_ == [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]


def testEnsembleSelectionRoundingLosses():
    """
    Makes sure ensemble selection rounding works as intended
    """

    ensemble = EnsembleSelection(
        ensemble_size=5,
        task_type=REGRESSION,
        random_state=0,
        metric=root_mean_squared_error,
        tie_breaker_default="random",
        use_best=False,
        tie_breaker_metric=None,
        round_losses=True,
    )

    # -- Case 1: no change due to low losses
    y_true = np.full((100), 0.99, dtype=np.float32)
    predictions = [
        np.full((100), 0.98999),
        np.full((100), 0.98999),
        np.full((100), 0.98999),
    ]
    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(3)])

    # Test if values where not rounded
    np.testing.assert_array_equal(
        np.array(
            [
                1.0009536743127432e-05,
                1.0009536743127432e-05,
                1.0009536743238458e-05,
                1.0009536743127432e-05,
                1.0009536743016413e-05,
            ]
        ),
        ensemble.trajectory_,
    )

    # -- Case 2: change in loss values due to rounding
    predictions = [
        np.full((100), 0.984),
        np.full((100), 0.982),
        np.full((100), 0.986),
        np.full((100), 0.988),
    ]
    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(4)])
    # Test if values where rounded
    np.testing.assert_array_equal(
        np.array([0.002, 0.002, 0.002, 0.002, 0.002]), ensemble.trajectory_
    )

    # -- Case 3: change in selection of base models due to rounding
    # Almost equal predictions, should be weighted almost equally
    predictions = [
        np.full((100), 0.9888499999998999999998999998999889),
        np.full((100), 0.9888499999999999999989999999899999),
    ]
    ensemble.fit(predictions, y_true, identifiers=[(i, i, i) for i in range(3)])
    np.testing.assert_array_equal(
        np.array([0.00115, 0.00115, 0.00115, 0.00115, 0.00115]), ensemble.trajectory_
    )
    assert ensemble.indices_ == [
        0,
        1,
        1,
        0,
        1,
    ]  # without rounding this would be [1, 1, 1, 1, 1]
    assert [0.4, 0.6]  # without rounding this would be [0. 1.]
