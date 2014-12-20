'''
Created on Dec 18, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

from AutoML2015.data.data_converter import convert_to_bin
from AutoML2015.models.evaluate import evaluate
from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.util import get_dataset
from HPOlibConfigSpace.random_sampler import RandomSampler

N_TEST_RUNS = 5


class Dummy(object):
    pass


class Test(unittest.TestCase):

    def test_evaluate_multiclass_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        D = Dummy()
        D.info = {'metric': 'bac_metric', 'task': 'multiclass.classification'}
        D.data = {'X_train': X_train, 'Y_train': Y_train}
        D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']

        configuration_space = AutoSklearnClassifier.\
            get_hyperparameter_search_space(multiclass=True)

        sampler = RandomSampler(configuration_space, 1)

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print "Evaluate configuration: " + str(i)
            configuration = sampler.sample_configuration()
            err[i] = evaluate(D, configuration)

            self.assertGreaterEqual(err[i], 0.0)

        print "Number of times it was worse than random guessing:" + str(np.sum(err > 1))

    def test_evaluate_multilabel_classification(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')
        Y_train = np.array(convert_to_bin(Y_train, 3))
        Y_test = np.array(convert_to_bin(Y_test, 3))

        D = Dummy()
        D.info = {'metric': 'f1_metric', 'task': 'multilabel.classification'}
        D.data = {'X_train': X_train, 'Y_train': Y_train}
        D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']

        configuration_space = AutoSklearnClassifier.\
            get_hyperparameter_search_space(multilabel=True)

        sampler = RandomSampler(configuration_space, 1)

        err = np.zeros([N_TEST_RUNS])
        for i in range(N_TEST_RUNS):
            print "Evaluate configuration: " + str(i)
            configuration = sampler.sample_configuration()
            err[i] = evaluate(D, configuration)

            self.assertGreaterEqual(err[i], 0.0)

        print "Number of times it was worse than random guessing:" + str(
            np.sum(err > 1))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_evaluate']
    unittest.main()
