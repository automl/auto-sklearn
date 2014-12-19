'''
Created on Dec 19, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

from AutoML2015.ensembles import weighted_ensemble
from AutoML2015.ensembles import ensemble_prediction

from AutoSklearn.util import get_dataset

N_TEST_RUNS = 10


class Test(unittest.TestCase):

    def test_weighted_ensemble(self):

        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        all_predictions = []
        for i in range(N_TEST_RUNS):
            predictions = np.load("/home/kleinaa/devel/git/automl2015/test_predictions/predictions_" + str(i) + ".npy")

            all_predictions.append(predictions)
            weights = weighted_ensemble(np.array(all_predictions), Y_train)

            self.assertEqual(weights.shape[0], i + 1)
            self.assertAlmostEqual(weights.sum(), 1.0)
            for w in weights:
                self.assertLessEqual(w, 1.0)
                self.assertGreaterEqual(w, 0.0)

    def test_ensemble_prediction(self):
        n_models = 10
        n_points = 20
        n_classes = 2
        pred = np.random.rand(n_models, n_points, n_classes)
        w = np.ones([n_models]) * (1.0 / float(n_models))
        p_hat = ensemble_prediction(pred, w)
        p = pred.mean(axis=0)

        for i in range(n_points):
            for j in range(n_classes):
                self.assertAlmostEqual(p[i, j], p_hat[i, j])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
