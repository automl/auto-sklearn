__author__ = 'feurerm'

import numpy as np
import time
import unittest

import itertools

import sklearn.datasets
import sklearn.decomposition

from AutoSklearn.autosklearn import AutoSklearnClassifier

class TestAllCombinations(unittest.TestCase):
    def get_iris(self):
        iris = sklearn.datasets.load_iris()
        X = iris.data
        Y = iris.target
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        rs.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        X_train = X[:100]
        Y_train = Y[:100]
        X_test = X[100:]
        Y_test = Y[100:]
        return X_train, Y_train, X_test, Y_test

    def test_all_combinations(self):
        # TODO: do the combination testing on the basis of one component
        # TODO: automate the testing, so far it is enumerated by hand
        parameter_combinations = list()

        libsvm_svc = []
        libsvm_svc_C_values = range(-5, 15 + 1)
        libsvm_svc_gamma_values = range(-15, 3 + 1)
        for C, gamma in itertools.product(libsvm_svc_C_values, libsvm_svc_gamma_values):
            libsvm_svc.append({"libsvm_svc:LOG2_C": C,
                               "libsvm_svc:LOG2_gamma": gamma,
                               "classifier": "libsvm_svc"})
        print "Parameter configurations LibSVM-SVC", len(libsvm_svc)

        liblinear = []
        liblinear_C_values = range(-5, 15 + 1)
        for C in liblinear_C_values:
            for penalty_and_loss in [{"penalty": "l1", "loss": "l2"},
                                     {"penalty": "l2", "loss": "l1"},
                                     {"penalty": "l2", "loss": "l2"}]:
                liblinear.append({"liblinear:LOG2_C": C,
                                "liblinear:penalty": penalty_and_loss["penalty"],
                                "liblinear:loss": penalty_and_loss["loss"],
                                "classifier": "liblinear"})
        print "Parameter configurations LibLinear", len(liblinear)

        random_forest = []
        random_forest_n_estimators = range(10, 100 + 1, 10)
        # This makes things too expensive
        # random_forst_min_samples_leaf = [1, 2, 4, 7, 10, 15, 20]
        random_forst_min_splits = [1, 2, 4, 7, 10]
        random_forest_max_features = np.linspace(0.01, 1.0, 8)
        random_forest_max_features = itertools.chain(
            random_forest_max_features, ["sqrt", "log2"])
        random_forest_criterion = ["gini", "entropy"]
        # random_forest_bootstrap = [True, False]

        #for n_est, min_leaf, min_splits, max_features, criterion, bootstrap in \
        for n_est, min_splits, max_features, criterion in \
            itertools.product(random_forest_n_estimators,
                              #random_forst_min_samples_leaf,
                              random_forst_min_splits,
                              random_forest_max_features,
                              random_forest_criterion):
                              #random_forest_bootstrap)
            random_forest.append(({"random_forest:n_estimators": n_est,
                                   "random_forest:criterion": criterion,
                                   "random_forest:max_features": max_features,
                                   "random_forest:min_samples_split": min_splits,
                                   #"random_forest:min_samples_leaf": min_leaf,
                                   #"random_forest:bootstrap": bootstrap,
                                   "classifier": "random_forest"}))
        print "Parameter configurations RF", len(random_forest)

        pca = []
        pca_n_components = np.linspace(0.60, 1.0, 10)
        # pca_whiten = [True, False]
        #for n_components, whiten in itertools.product(pca_n_components):
                                                      #pca_whiten):
        for n_components in pca_n_components:
            pca.append({"pca:n_components": n_components,
                        #"pca:whiten": whiten,
                        "preprocessor": "pca"})
        print "Parameter configurations PCA", len(pca)

        classifiers = [liblinear, libsvm_svc, random_forest]
        preprocessors = [pca, [{"preprocessor": None}]]

        for classifier, preprocessor in itertools.product(classifiers,
                                                          preprocessors):
            print classifier[0]["classifier"], preprocessor[0]["preprocessor"]
            for classifier_params, preprocessor_params in itertools.product(
                    classifier, preprocessor):
                params = {}
                params.update(classifier_params)
                params.update(preprocessor_params)
                parameter_combinations.append(params)

        starttime = time.time()
        print len(parameter_combinations)
        for i, parameter_combination in enumerate(parameter_combinations):
            auto = AutoSklearnClassifier(parameters=parameter_combination)
            X_train, Y_train, X_test, Y_test = self.get_iris()
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(X_test)
            accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)

            if i % 1000 == 0 and i != 0:
                print "Iteration", i
                print (time.time() - starttime) * 1000 / i

        print "Finished, took", time.time() - starttime

