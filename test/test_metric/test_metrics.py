import unittest

import pytest

import numpy as np
import sklearn.metrics

import autosklearn.metrics

from autosklearn.metrics import calculate_score, calculate_loss, calculate_metric
from autosklearn.constants import BINARY_CLASSIFICATION, REGRESSION

from smac.utils.constants import MAXINT


class TestScorer(unittest.TestCase):

    def test_predict_scorer_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._PredictScorer(
            'bac', sklearn.metrics.balanced_accuracy_score,
            1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._PredictScorer(
            name='accuracy', score_func=sklearn.metrics.accuracy_score,
            optimum=1, worst_possible_result=0, sign=-1, kwargs={})

        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_multiclass(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        y_pred = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        scorer = autosklearn.metrics._PredictScorer(
            'bac', sklearn.metrics.balanced_accuracy_score,
            1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 0, -1, {})

        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.25)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.25)

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 0, -1, {})

        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_regression(self):
        y_true = np.arange(0, 1.01, 0.1)
        y_pred = y_true.copy()

        scorer = autosklearn.metrics._PredictScorer(
            'r2', sklearn.metrics.r2_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.ones(y_true.shape) * np.mean(y_true)
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

    def test_proba_scorer_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

        y_pred = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        y_pred = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, -1, {})

        y_pred = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -0.69314718055994529)

    def test_proba_scorer_multiclass(self):
        y_true = [0, 1, 2]
        y_pred = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

        y_pred = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0986122886681098)

        y_pred = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0986122886681096)

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, -1, {})

        y_pred = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0986122886681096)

    def test_proba_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.34657359027997314)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        scorer = autosklearn.metrics._ProbaScorer(
            'log_loss', sklearn.metrics.log_loss, 0, MAXINT, -1, {})

        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -0.34657359027997314)

    def test_threshold_scorer_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        scorer = autosklearn.metrics._ThresholdScorer(
            'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._ThresholdScorer(
            'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, -1, {})

        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_threshold_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._ThresholdScorer(
            'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._ThresholdScorer(
            'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, -1, {})

        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_sign_flip(self):
        y_true = np.arange(0, 1.01, 0.1)
        y_pred = y_true.copy()

        scorer = autosklearn.metrics.make_scorer(
            'r2', sklearn.metrics.r2_score, greater_is_better=True)

        score = scorer(y_true, y_pred + 1.0)
        self.assertAlmostEqual(score, -9.0)

        score = scorer(y_true, y_pred + 0.5)
        self.assertAlmostEqual(score, -1.5)

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        scorer = autosklearn.metrics.make_scorer(
            'r2', sklearn.metrics.r2_score, greater_is_better=False)

        score = scorer(y_true, y_pred + 1.0)
        self.assertAlmostEqual(score, 9.0)

        score = scorer(y_true, y_pred + 0.5)
        self.assertAlmostEqual(score, 1.5)

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)


class TestMetricsDoNotAlterInput(unittest.TestCase):

    def test_regression_metrics(self):
        for metric, scorer in autosklearn.metrics.REGRESSION_METRICS.items():
            y_true = np.random.random(100).reshape((-1, 1))
            y_pred = y_true.copy() + np.random.randn(100, 1) * 0.1

            if metric == 'mean_squared_log_error':
                y_true = np.abs(y_true)
                y_pred = np.abs(y_pred)

            y_true_2 = y_true.copy()
            y_pred_2 = y_pred.copy()
            self.assertTrue(np.isfinite(scorer(y_true_2, y_pred_2)))
            np.testing.assert_array_almost_equal(y_true, y_true_2,
                                                 err_msg=metric)
            np.testing.assert_array_almost_equal(y_pred, y_pred_2,
                                                 err_msg=metric)

    def test_classification_metrics(self):
        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            y_true = np.random.randint(0, 2, size=(100, 1))
            y_pred = np.random.random(200).reshape((-1, 2))
            y_pred = np.array([y_pred[i] / np.sum(y_pred[i])
                               for i in range(100)])

            y_true_2 = y_true.copy()
            y_pred_2 = y_pred.copy()
            try:
                self.assertTrue(np.isfinite(scorer(y_true_2, y_pred_2)))
                np.testing.assert_array_almost_equal(y_true, y_true_2,
                                                     err_msg=metric)
                np.testing.assert_array_almost_equal(y_pred, y_pred_2,
                                                     err_msg=metric)
            except ValueError as e:
                if e.args[0] == 'Samplewise metrics are not available outside' \
                        ' of multilabel classification.':
                    pass
                else:
                    raise e


class TestMetric(unittest.TestCase):

    def test_regression_all(self):

        for metric, scorer in autosklearn.metrics.REGRESSION_METRICS.items():
            y_true = np.array([1, 2, 3, 4])
            y_pred = y_true.copy()
            previous_score = scorer._optimum
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([3, 4, 5, 6])
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            if scorer.name == 'mean_squared_log_error':
                continue

            y_pred = np.array([-1, 0, -1, 0])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([-5, 10, 7, -3])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

    def test_classification_binary(self):

        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            # Skip functions not applicable for binary classification.
            # TODO: Average precision should work for binary classification,
            # TODO: but its behavior is not right. When y_pred is completely
            # TODO: wrong, it does return 0.5, but when it is not completely
            # TODO: wrong, it returns value smaller than 0.5.
            if metric in ['average_precision',
                          'precision_samples', 'recall_samples', 'f1_samples']:
                continue

            y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            y_pred = \
                np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
            previous_score = scorer._optimum
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = \
                np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = \
                np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = \
                np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

    def test_classification_multiclass(self):

        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            # Skip functions not applicable for multiclass classification.
            if metric in ['roc_auc', 'average_precision',
                          'precision', 'recall', 'f1', 'precision_samples',
                          'recall_samples', 'f1_samples']:
                continue
            y_true = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
            y_pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            previous_score = scorer._optimum
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            # less labels in the targets than in the predictions
            y_true = np.array([0.0, 0.0, 1.0, 1.0])
            y_pred = np.array([
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            score = scorer(y_true, y_pred)
            self.assertTrue(np.isfinite(score))

    def test_classification_multilabel(self):

        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            # Skip functions not applicable for multi-label classification.
            if metric in ['roc_auc', 'log_loss',
                          'precision', 'recall', 'f1', 'balanced_accuracy']:
                continue
            y_true = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
            y_pred = y_true.copy()
            previous_score = scorer._optimum
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)


class TestCalculateScore(unittest.TestCase):

    def test_unsupported_task_type(self):
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = \
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        scorer = autosklearn.metrics.accuracy

        raised = False
        try:
            calculate_score(y_true, y_pred, 6, scorer)
        except NotImplementedError:
            raised = True
        self.assertTrue(raised)

    def test_classification_scoring_functions(self):

        scoring_functions = list(autosklearn.metrics.CLASSIFICATION_METRICS.values())
        scoring_functions.remove(autosklearn.metrics.accuracy)
        fail_metrics = ['precision_samples', 'recall_samples', 'f1_samples']
        success_metrics = list(autosklearn.metrics.CLASSIFICATION_METRICS.keys())
        for metric in fail_metrics:
            success_metrics.remove(metric)

        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = \
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        score_dict = calculate_score(y_true, y_pred, BINARY_CLASSIFICATION,
                                     autosklearn.metrics.accuracy,
                                     scoring_functions)

        self.assertIsInstance(score_dict, dict)
        self.assertTrue(len(success_metrics), len(score_dict))
        for metric in fail_metrics:
            self.assertNotIn(metric, score_dict.keys())
        for metric in success_metrics:
            self.assertIn(metric, score_dict.keys())
            self.assertAlmostEqual(autosklearn.metrics.CLASSIFICATION_METRICS[metric]._optimum,
                                   score_dict[metric])

    def test_regression_scoring_functions(self):

        scoring_functions = list(autosklearn.metrics.REGRESSION_METRICS.values())
        scoring_functions.remove(autosklearn.metrics.root_mean_squared_error)

        metrics = list(autosklearn.metrics.REGRESSION_METRICS.keys())
        metrics.remove('mean_squared_log_error')

        y_true = np.array([1, 2, 3, -4])
        y_pred = y_true.copy()

        score_dict = calculate_score(y_true, y_pred, REGRESSION,
                                     autosklearn.metrics.root_mean_squared_error,
                                     scoring_functions)

        self.assertIsInstance(score_dict, dict)
        self.assertTrue(len(metrics), len(score_dict))
        for metric in metrics:
            self.assertIn(metric, score_dict.keys())
            self.assertAlmostEqual(autosklearn.metrics.REGRESSION_METRICS[metric]._optimum,
                                   score_dict[metric])

    def test_classification_only_metric(self):
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = \
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        scorer = autosklearn.metrics.accuracy

        score = calculate_score(y_true, y_pred, BINARY_CLASSIFICATION, scorer)

        previous_score = scorer._optimum
        self.assertAlmostEqual(score, previous_score)

    def test_regression_only_metric(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = y_true.copy()
        scorer = autosklearn.metrics.root_mean_squared_error

        score = calculate_score(y_true, y_pred, REGRESSION, scorer)
        previous_score = scorer._optimum
        self.assertAlmostEqual(score, previous_score)


def test_calculate_loss():
    # In a 0-1 ranged scorer, make sure that the loss
    # has an expected positive value
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    assert pytest.approx(score) == calculate_score(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
    )
    assert pytest.approx(1.0 - score) == calculate_loss(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
    )

    # Test the dictionary case
    score_dict = calculate_score(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
        scoring_functions=[autosklearn.metrics.accuracy, autosklearn.metrics.balanced_accuracy]
    )
    expected_score_dict = {
        'accuracy': 0.9,
        'balanced_accuracy': 0.9285714285714286,
    }
    loss_dict = calculate_loss(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
        scoring_functions=[autosklearn.metrics.accuracy, autosklearn.metrics.balanced_accuracy]
    )
    for expected_metric, expected_score in expected_score_dict.items():
        assert pytest.approx(expected_score) == score_dict[expected_metric]
        assert pytest.approx(1-expected_score) == loss_dict[expected_metric]

    # Lastly make sure that metrics whose optimum is zero
    # are also properly working
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y_pred = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    assert pytest.approx(0 - score) == calculate_score(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metric=autosklearn.metrics.mean_squared_error,
    )
    assert pytest.approx(score) == calculate_loss(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metric=autosklearn.metrics.mean_squared_error,
    )


def test_calculate_metric():
    # metric to be maximized
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    assert pytest.approx(score) == calculate_metric(
        solution=y_true,
        prediction=y_pred,
        task_type=BINARY_CLASSIFICATION,
        metric=autosklearn.metrics.accuracy,
    )

    # metric to be minimized
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y_pred = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    assert pytest.approx(score) == calculate_metric(
        solution=y_true,
        prediction=y_pred,
        task_type=REGRESSION,
        metric=autosklearn.metrics.mean_squared_error,
    )
