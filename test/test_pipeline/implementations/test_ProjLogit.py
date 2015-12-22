import unittest
import os
import numpy as np
#import scipy.io

from autosklearn.pipeline.implementations.ProjLogit import ProjLogit


class TestProjLogit(unittest.TestCase):
    def test_sparse_filtering(self):
        """Test logistic regression implementation based on least squares"""

        # simple test that should work out
        trainx = np.random.rand(100,3)
        trainy = np.zeros(10000)
        testx = np.random.rand(100,3)
        testy = np.zeros(100)
        for i in range(100):
            if trainx[i, 2] > 0.5:
                trainy[i] = 1
        for i in range(100):
            if testx[i, 2] > 0.5:
                testy[i] = 1

        model = ProjLogit(max_epochs = 10, verbose = True)
        model.fit(trainx, trainy)
        print("weights 0:")
        print(model.w0)
        predicted_prob = model.predict_proba(testx)
        predicted2 = np.argmax(predicted_prob, axis = 1)
        predicted = model.predict(testx)

        #print(predicted)
        #print(testy)
        #print((predicted != testy).sum())
        #print((predicted2 != testy).sum())
        self.assertTrue((predicted == predicted2).all())
        self.assertTrue(((1 - predicted_prob.sum(axis=1)) < 1e-3).all())
        self.assertTrue((predicted != testy).sum() < 20)
