'''
Created on Dec 18, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np
import sklearn.datasets

from AutoML2015.data.data_manager import DataManager
from AutoML2015.models.evaluate import evaluate
from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.util import get_dataset
from HPOlibConfigSpace.random_sampler import RandomSampler

N_TEST_RUNS = 100


class Dummy(object):
    pass


class Test(unittest.TestCase):

    def test_evaluate(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        D = Dummy()
        D.info = {'metric': 'bac_metric', 'task': 'multiclass.classification'}
        D.data = {'X_train': X_train, 'Y_train': Y_train}

        configuration_space = AutoSklearnClassifier.get_hyperparameter_search_space()

        sampler = RandomSampler(configuration_space, 1)

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print "Evaluate configuration: " + str(i)
            configuration = sampler.sample_configuration()
            err[i] = evaluate(D, configuration)

            self.assertGreaterEqual(err[i], 0.0)

        print "Number of times it was worse than random guessing:" + str(np.sum(err > 1))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_evaluate']
    unittest.main()
