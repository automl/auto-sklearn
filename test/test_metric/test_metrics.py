import warnings

import numpy as np
import sklearn.metrics

import autosklearn.metrics
from autosklearn.constants import BINARY_CLASSIFICATION, REGRESSION
from autosklearn.metrics import (
    calculate_losses,
    calculate_scores,
    compute_single_metric,
)

import pytest
import unittest


class TestScorer(unittest.TestCase):
    def test_needs_X(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        def dummy_metric(y_true, y_pred, X_data=None, **kwargs):
            if not np.array_equal(np.array([45]), X_data):
                raise ValueError(f"is {X_data}")
            return 1

        scorer = autosklearn.metrics._PredictScorer(
            "accuracy", dummy_metric, 1, 0, 1, {}, needs_X=True
        )
        scorer(y_true, y_pred, X_data=np.array([45]))

        scorer_nox = autosklearn.metrics._PredictScorer(
            "accuracy", dummy_metric, 1, 0, 1, {}, needs_X=False
        )
        with self.assertRaises(ValueError) as cm:
            scorer_nox(y_true, y_pred, X_data=np.array([32]))
        the_exception = cm.exception
        # X_data is not forwarded
        self.assertEqual(the_exception.args[0], "is None")

        scorer_nox = autosklearn.metrics._PredictScorer(
            "accuracy", sklearn.metrics.accuracy_score, 1, 0, 1, {}, needs_X=False
        )
        scorer_nox(y_true, y_pred, X_data=np.array([32]))


@pytest.mark.parametrize(
    "y_pred, y_true, scorer, expected_score",
    [
        (
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.accuracy,
            1.0,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.accuracy,
            0.5,
        ),
        (
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.balanced_accuracy,
            0.5,
        ),
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([0, 1, 2]),
            autosklearn.metrics.accuracy,
            1.0,
        ),
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([0, 1, 2]),
            autosklearn.metrics.accuracy,
            0.333333333,
        ),
        (
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            np.array([0, 1, 2]),
            autosklearn.metrics.accuracy,
            0.333333333,
        ),
        (
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            np.array([0, 1, 2]),
            autosklearn.metrics.balanced_accuracy,
            0.333333333,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.accuracy,
            1.0,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.accuracy,
            0.25,
        ),
        (
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.accuracy,
            0.25,
        ),
        (
            np.arange(0, 1.01, 0.1),
            np.arange(0, 1.01, 0.1),
            autosklearn.metrics.r2,
            1.0,
        ),
        (
            np.ones(np.arange(0, 1.01, 0.1).shape) * np.mean(np.arange(0, 1.01, 0.1)),
            np.arange(0, 1.01, 0.1),
            autosklearn.metrics.r2,
            0.0,
        ),
        (
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.log_loss,
            0.0,
        ),
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([0, 1, 2]),
            autosklearn.metrics.log_loss,
            0.0,
        ),
        (
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.roc_auc,
            1.0,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.roc_auc,
            0.5,
        ),
        (
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            np.array([0, 0, 1, 1]),
            autosklearn.metrics.roc_auc,
            0.5,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.roc_auc,
            1.0,
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.roc_auc,
            0.5,
        ),
        (
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autosklearn.metrics.roc_auc,
            0.5,
        ),
    ],
)
def test_scorer(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    scorer: autosklearn.metrics.Scorer,
    expected_score: float,
) -> None:
    """
    Expects
    -------
    * Expected scores are equal to scores gained from implementing assembled scorers.
    """
    result_score = scorer(y_true, y_pred)
    assert expected_score == pytest.approx(result_score)


@pytest.mark.parametrize(
    "y_pred, y_true, expected_score",
    [
        (
            np.arange(0, 1.01, 0.1) + 1.0,
            np.arange(0, 1.01, 0.1),
            -9.0,
        ),
        (
            np.arange(0, 1.01, 0.1) + 0.5,
            np.arange(0, 1.01, 0.1),
            -1.5,
        ),
        (
            np.arange(0, 1.01, 0.1),
            np.arange(0, 1.01, 0.1),
            1.0,
        ),
    ],
)
def test_sign_flip(
    y_pred: np.array,
    y_true: np.array,
    expected_score: float,
) -> None:
    """
    Expects
    -------
    * Flipping greater_is_better for r2_score result in flipped signs of its output.
    """
    greater_true_scorer = autosklearn.metrics.make_scorer(
        "r2", sklearn.metrics.r2_score, greater_is_better=True
    )
    greater_true_score = greater_true_scorer(y_true, y_pred)
    assert expected_score == pytest.approx(greater_true_score)

    greater_false_scorer = autosklearn.metrics.make_scorer(
        "r2", sklearn.metrics.r2_score, greater_is_better=False
    )
    greater_false_score = greater_false_scorer(y_true, y_pred)
    assert (expected_score * -1.0) == pytest.approx(greater_false_score)


def test_regression_metrics():
    """
    Expects
    -------
    * Test metrics do not change output for autosklearn.metrics.REGRESSION_METRICS.
    """
    for metric, scorer in autosklearn.metrics.REGRESSION_METRICS.items():
        y_true = np.random.random(100).reshape((-1, 1))
        y_pred = y_true.copy() + np.random.randn(100, 1) * 0.1

        if metric == "mean_squared_log_error":
            y_true = np.abs(y_true)
            y_pred = np.abs(y_pred)

        y_true_2 = y_true.copy()
        y_pred_2 = y_pred.copy()
        assert np.isfinite(scorer(y_true_2, y_pred_2))
        np.testing.assert_array_almost_equal(y_true, y_true_2, err_msg=metric)
        np.testing.assert_array_almost_equal(y_pred, y_pred_2, err_msg=metric)


def test_classification_metrics():
    """
    Expects
    -------
    * Test metrics do not change output for autosklearn.metrics.CLASSIFICATION_METRICS.
    """
    for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
        y_true = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.random.random(200).reshape((-1, 2))
        y_pred = np.array([y_pred[i] / np.sum(y_pred[i]) for i in range(100)])

        y_true_2 = y_true.copy()
        y_pred_2 = y_pred.copy()
        try:
            assert np.isfinite(scorer(y_true_2, y_pred_2))
            np.testing.assert_array_almost_equal(y_true, y_true_2, err_msg=metric)
            np.testing.assert_array_almost_equal(y_pred, y_pred_2, err_msg=metric)
        except ValueError as e:
            if (
                e.args[0] == "Samplewise metrics are not available outside"
                " of multilabel classification."
            ):
                pass
            else:
                raise e


def test_regression_all():
    """
    Expects
    -------
    * Correct scores from REGRESSION_METRICS.
    """
    for metric, scorer in autosklearn.metrics.REGRESSION_METRICS.items():
        if scorer.name == "mean_squared_log_error":
            continue

        y_true = np.array([1, 2, 3, 4])

        y_pred_list = [
            np.array([1, 2, 3, 4]),
            np.array([3, 4, 5, 6]),
            np.array([-1, 0, -1, 0]),
            np.array([-5, 10, 7, -3]),
        ]

        score_list = [scorer(y_true, y_pred) for y_pred in y_pred_list]

        assert scorer._optimum == pytest.approx(score_list[0])
        assert score_list == sorted(score_list, reverse=True)


def test_classification_binary():
    """
    Expects
    -------
    * Correct scores from CLASSIFICATION_METRICS for binary classification.
    """
    for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
        # Skip functions not applicable for binary classification.
        # TODO: Average precision should work for binary classification,
        # TODO: but its behavior is not right. When y_pred is completely
        # TODO: wrong, it does return 0.5, but when it is not completely
        # TODO: wrong, it returns value smaller than 0.5.
        if metric in [
            "average_precision",
            "precision_samples",
            "recall_samples",
            "f1_samples",
        ]:
            continue

        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        y_pred_list = [
            np.array(
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
            ),
            np.array(
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
            ),
            np.array(
                [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            ),
            np.array(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            ),
        ]

        score_list = [scorer(y_true, y_pred) for y_pred in y_pred_list]

        assert scorer._optimum == pytest.approx(score_list[0])
        assert score_list == sorted(score_list, reverse=True)


def test_classification_multiclass():
    """
    Expects
    -------
    * Correct scores from CLASSIFICATION_METRICS for multiclass classification.
    """
    # The last check in this test has a mismatch between the number of
    # labels predicted in y_pred and the number of labels in y_true.
    # This triggers several warnings but we are aware.
    #
    # TODO convert to pytest with fixture
    #
    #   This test should be parameterized so we can identify which metrics
    #   cause which warning specifically and rectify if needed.
    ignored_warnings = [(UserWarning, "y_pred contains classes not in y_true")]

    for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
        # Skip functions not applicable for multiclass classification.
        if metric in [
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
            "precision_samples",
            "recall_samples",
            "f1_samples",
        ]:
            continue

        y_true = np.array([0.0, 0.0, 1.0, 1.0, 2.0])

        y_pred_list = [
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ]

        score_list = [scorer(y_true, y_pred) for y_pred in y_pred_list]

        assert scorer._optimum == pytest.approx(score_list[0])
        assert score_list == sorted(score_list, reverse=True)

        # less labels in the targets than in the predictions
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        with warnings.catch_warnings():
            for category, message in ignored_warnings:
                warnings.filterwarnings("ignore", category=category, message=message)

            score = scorer(y_true, y_pred)
            assert np.isfinite(score)


def test_classification_multilabel():
    """
    Expects
    -------
    * Correct scores from CLASSIFICATION_METRICS for multi-label classification.
    """
    for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
        # Skip functions not applicable for multi-label classification.
        if metric in [
            "roc_auc",
            "log_loss",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ]:
            continue
        y_true = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])

        y_pred_list = [
            np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]),
            np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]]),
            np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]]),
        ]

        score_list = [scorer(y_true, y_pred) for y_pred in y_pred_list]

        assert scorer._optimum == pytest.approx(score_list[0])
        assert score_list == sorted(score_list, reverse=True)


class TestCalculateScore(unittest.TestCase):
    def test_unsupported_task_type(self):
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = np.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        scorer = autosklearn.metrics.accuracy

        raised = False
        try:
            calculate_scores(y_true, y_pred, 6, scorer)
        except NotImplementedError:
            raised = True
        self.assertTrue(raised)

    def test_classification_scoring_functions(self):

        scoring_functions = list(autosklearn.metrics.CLASSIFICATION_METRICS.values())
        scoring_functions.remove(autosklearn.metrics.accuracy)
        fail_metrics = ["precision_samples", "recall_samples", "f1_samples"]
        success_metrics = list(autosklearn.metrics.CLASSIFICATION_METRICS.keys())
        for metric in fail_metrics:
            success_metrics.remove(metric)

        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = np.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        score_dict = calculate_scores(
            y_true,
            y_pred,
            BINARY_CLASSIFICATION,
            [autosklearn.metrics.accuracy],
            scoring_functions=scoring_functions,
        )

        self.assertIsInstance(score_dict, dict)
        self.assertTrue(len(success_metrics), len(score_dict))
        for metric in fail_metrics:
            self.assertNotIn(metric, score_dict.keys())
        for metric in success_metrics:
            self.assertIn(metric, score_dict.keys())
            self.assertAlmostEqual(
                autosklearn.metrics.CLASSIFICATION_METRICS[metric]._optimum,
                score_dict[metric],
            )

    def test_regression_scoring_functions(self):

        scoring_functions = list(autosklearn.metrics.REGRESSION_METRICS.values())
        scoring_functions.remove(autosklearn.metrics.root_mean_squared_error)

        metrics = list(autosklearn.metrics.REGRESSION_METRICS.keys())
        metrics.remove("mean_squared_log_error")

        y_true = np.array([1, 2, 3, -4])
        y_pred = y_true.copy()

        score_dict = calculate_scores(
            y_true,
            y_pred,
            REGRESSION,
            [autosklearn.metrics.root_mean_squared_error],
            scoring_functions=scoring_functions,
        )

        self.assertIsInstance(score_dict, dict)
        self.assertTrue(len(metrics), len(score_dict))
        for metric in metrics:
            self.assertIn(metric, score_dict.keys())
            self.assertAlmostEqual(
                autosklearn.metrics.REGRESSION_METRICS[metric]._optimum,
                score_dict[metric],
            )

    def test_classification_only_metric(self):
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = np.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        scorer = autosklearn.metrics.accuracy

        score = calculate_scores(y_true, y_pred, BINARY_CLASSIFICATION, [scorer])[
            "accuracy"
        ]

        previous_score = scorer._optimum
        self.assertAlmostEqual(score, previous_score)

    def test_regression_only_metric(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = y_true.copy()
        scorer = autosklearn.metrics.root_mean_squared_error

        score = calculate_scores(y_true, y_pred, REGRESSION, [scorer])[
            "root_mean_squared_error"
        ]
        previous_score = scorer._optimum
        self.assertAlmostEqual(score, previous_score)


def test_calculate_losses():
    # In a 0-1 ranged scorer, make sure that the loss
    # has an expected positive value
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    assert {"accuracy": pytest.approx(score)} == calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
    )
    assert {"accuracy": pytest.approx(1.0 - score)} == calculate_losses(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
    )

    # Test two metrics
    score_dict = calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[
            autosklearn.metrics.accuracy,
            autosklearn.metrics.balanced_accuracy,
        ],
    )
    expected_score_dict = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.9285714285714286,
    }
    loss_dict = calculate_losses(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[
            autosklearn.metrics.accuracy,
            autosklearn.metrics.balanced_accuracy,
        ],
    )
    for expected_metric, expected_score in expected_score_dict.items():
        assert pytest.approx(expected_score) == score_dict[expected_metric]
        assert pytest.approx(1 - expected_score) == loss_dict[expected_metric]

    # Test no metric
    with pytest.raises(
        ValueError, match="Number of metrics to compute must be greater than zero."
    ):
        calculate_scores(
            solution=y_true,
            prediction=y_pred,
            task_type=BINARY_CLASSIFICATION,
            metrics=[],
        )

    with pytest.raises(
        ValueError, match="Number of metrics to compute must be greater than zero."
    ):
        calculate_scores(
            solution=y_true,
            prediction=y_pred,
            task_type=BINARY_CLASSIFICATION,
            metrics=[],
            scoring_functions=[
                autosklearn.metrics.accuracy,
                autosklearn.metrics.balanced_accuracy,
            ],
        )

    # Test the same metric twice
    accuracy_fixture = {"accuracy": pytest.approx(0.9)}
    assert accuracy_fixture == calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy, autosklearn.metrics.accuracy],
    )
    assert accuracy_fixture == calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
        scoring_functions=[autosklearn.metrics.accuracy],
    )
    assert accuracy_fixture == calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
        scoring_functions=[autosklearn.metrics.accuracy, autosklearn.metrics.accuracy],
    )

    # Test the same name for multiple metrics!
    bogus_accuracy = autosklearn.metrics.make_scorer(
        "accuracy",
        score_func=sklearn.metrics.roc_auc_score,
    )
    with pytest.raises(ValueError, match="used multiple times"):
        calculate_scores(
            solution=y_true,
            prediction=y_pred,
            task_type=BINARY_CLASSIFICATION,
            metrics=[autosklearn.metrics.accuracy, bogus_accuracy],
        )

    # Test additional scoring functions
    score_dict = calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
        scoring_functions=[
            autosklearn.metrics.accuracy,
            autosklearn.metrics.balanced_accuracy,
        ],
    )
    expected_score_dict = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.9285714285714286,
    }
    loss_dict = calculate_losses(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metrics=[autosklearn.metrics.accuracy],
        scoring_functions=[
            autosklearn.metrics.accuracy,
            autosklearn.metrics.balanced_accuracy,
        ],
    )
    for expected_metric, expected_score in expected_score_dict.items():
        assert pytest.approx(expected_score) == score_dict[expected_metric]
        assert pytest.approx(1 - expected_score) == loss_dict[expected_metric]

    # Lastly make sure that metrics whose optimum is zero
    # are also properly working
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y_pred = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    assert {"mean_squared_error": pytest.approx(0 - score)} == calculate_scores(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metrics=[autosklearn.metrics.mean_squared_error],
    )
    assert {"mean_squared_error": pytest.approx(score)} == calculate_losses(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metrics=[autosklearn.metrics.mean_squared_error],
    )


def test_calculate_metric():
    # metric to be maximized
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    assert pytest.approx(score) == compute_single_metric(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
    )

    # metric to be minimized
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y_pred = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    assert pytest.approx(score) == compute_single_metric(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metric=autosklearn.metrics.mean_squared_error,
    )
