'''
Created on Dec 19, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

import autosklearn.ensemble_script

N_TEST_RUNS = 10

class Test(unittest.TestCase):

    def setUp(self):
        self.n_models = 10
        self.n_points = 20
        self.n_classes = 5

    def test_weighted_ensemble(self):

        predictions = np.random.rand(self.n_models, self.n_points, self.n_classes)

        true_labels = np.random.randint(self.n_classes, size=self.n_points)

        weights = np.random.rand(self.n_models)

        weights /= weights.sum()

        metric = "f1_metric"
        task_type = "multiclass.classification"

        weights = autosklearn.ensemble_script.weighted_ensemble(predictions, true_labels, task_type, metric, weights)

        self.assertEqual(weights.shape[0], self.n_models)

        self.assertAlmostEqual(weights.sum(), 1.0)

        for w in weights:
                self.assertLessEqual(w, 1.0)
                self.assertGreaterEqual(w, 0.0)

    def test_ensemble_prediction(self):

        pred = np.random.rand(self.n_models, self.n_points, self.n_classes)
        w = np.ones([self.n_models]) * (1.0 / float(self.n_models))
        p_hat = autosklearn.ensemble_script.ensemble_prediction(pred, w)
        p = pred.mean(axis=0)

        # First test case
        for i in range(self.n_points):
            for j in range(self.n_classes):
                self.assertAlmostEqual(p[i, j], p_hat[i, j])

        # Second test case
        w = np.zeros([self.n_models])
        w[0] = 1.0
        p_hat = autosklearn.ensemble_script.ensemble_prediction(pred, w)
        for i in range(self.n_points):
            for j in range(self.n_classes):
                self.assertAlmostEqual(pred[0, i, j], p_hat[i, j])

    def test_weighted_ensemble_error(self):
        predictions = np.random.rand(self.n_models, self.n_points, self.n_classes)

        true_labels = np.random.randint(self.n_classes, size=self.n_points)

        weights = np.random.randn(self.n_models)
        weights /= weights.sum()

        metric = "f1_metric"
        task_type = "multiclass.classification"

        err = autosklearn.ensemble_script.weighted_ensemble_error(weights, predictions, true_labels, metric, task_type)

        self.assertAlmostEqual(err, 1.0, delta=0.3)

if __name__ == "__main__":
    unittest.main()
