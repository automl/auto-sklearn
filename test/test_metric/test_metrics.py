import unittest

import numpy as np
import sklearn.metrics

import autosklearn.metrics.classification_metrics



class TestScorer(unittest.TestCase):

    def test_predict_scorer_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._PredictScorer(
            'bac', autosklearn.metrics.classification_metrics.balanced_accuracy,
            1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, -1, {})

        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_multiclass(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        y_pred = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        scorer = autosklearn.metrics._PredictScorer(
            'bac', autosklearn.metrics.classification_metrics.balanced_accuracy,
            1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.333333333)

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, -1, {})

        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.25)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.25)

        scorer = autosklearn.metrics._PredictScorer(
            'bac', autosklearn.metrics.classification_metrics.balanced_accuracy,
            1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._PredictScorer(
            'accuracy', sklearn.metrics.accuracy_score, 1, -1, {})

        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_predict_scorer_regression(self):
        y_true = np.arange(0, 1.01, 0.1)
        y_pred = y_true.copy()

        scorer = autosklearn.metrics._PredictScorer(
            'r2', sklearn.metrics.r2_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.ones(y_true.shape) * np.mean(y_true)
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

    def test_proba_scorer_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

        y_pred = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        y_pred = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 0, -1, {})

        y_pred = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -0.69314718055994529)

    def test_proba_scorer_multiclass(self):
        y_true = [0, 1, 2]
        y_pred = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 0, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0)

        y_pred = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0986122886681098)

        y_pred = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0986122886681096)

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 0, -1, {})

        y_pred = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0986122886681096)

    def test_proba_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.34657359027997314)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.69314718055994529)

        scorer = autosklearn.metrics._ProbaScorer(
            'accuracy', sklearn.metrics.log_loss, 1, -1, {})

        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -0.34657359027997314)

    def test_threshold_scorer_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        scorer = autosklearn.metrics._ThresholdScorer(
            'accuracy', sklearn.metrics.roc_auc_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._ThresholdScorer(
            'accuracy', sklearn.metrics.roc_auc_score, 1, -1, {})

        y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, -1.0)

    def test_threshold_scorer_multilabel(self):
        y_true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        scorer = autosklearn.metrics._ThresholdScorer(
            'accuracy', sklearn.metrics.roc_auc_score, 1, 1, {})

        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        score = scorer(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5)

        scorer = autosklearn.metrics._ThresholdScorer(
            'accuracy', sklearn.metrics.roc_auc_score, 1, -1, {})

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
                if e.args[0] == 'Sample-based precision, recall, fscore is ' \
                                'not meaningful outside multilabel ' \
                                'classification. See the accuracy_score instead.':
                    pass
                else:
                    raise e


class TestMetric(unittest.TestCase):

    def test_regression_all(self):

        for metric, scorer in autosklearn.metrics.REGRESSION_METRICS.items():
            y_true = np.array([1, 2, 3, 4])
            y_pred = y_true.copy()

            # the best possible score of r2 loss is 1.
            if metric == 'r2':
                previous_score = 1
            else:
                previous_score = 0
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([3, 4, 5, 6])
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

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
            if metric in ['pac_score', 'precision_samples', 'recall_samples', 'f1_samples']:
                continue

            y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
            y_pred = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
            if metric is 'log_loss':
                previous_score = 0      # the best value for log loss is 0.
            else:
                previous_score = 1     # the best value for other losses is 1.
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            if metric is 'recall':
                y_pred = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
            else:
                y_pred = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

    def test_classification_multiclass(self):

        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            # Skip functions not applicable for multiclass classification.
            if metric in ['pac_score', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1',
                          'precision_samples', 'recall_samples', 'f1_samples']:
                continue
            y_true = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
            y_pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            if metric is 'log_loss': # the best possible score for log_loss is 0.
                previous_score = 0
            else:
                previous_score = 1 # the best value for other losses is 1, and we flip the sign to minimize.
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)


    def test_classification_multilabel(self):

        for metric, scorer in autosklearn.metrics.CLASSIFICATION_METRICS.items():
            # Skip functions not applicable for multi-label classification.
            if metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'average_precision', 'log_loss',
                          'pac_score', 'precision', 'recall', 'f1']:
                continue
            y_true = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
            y_pred = y_true.copy()
            previous_score = 1
            current_score = scorer(y_true, y_pred)
            self.assertAlmostEqual(current_score, previous_score)

            y_pred = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)

            y_pred = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]])
            previous_score = current_score
            current_score = scorer(y_true, y_pred)
            self.assertLess(current_score, previous_score)


if __name__=="__main__":
    unittest.main()