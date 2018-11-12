__author__ = 'feurerm'

import copy
import unittest

import numpy as np
import sklearn.datasets
import sklearn.metrics

from autosklearn.pipeline.components.data_preprocessing.balancing.balancing \
    import Balancing
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier
from autosklearn.pipeline.components.classification.gradient_boosting import GradientBoostingClassifier
from autosklearn.pipeline.components.classification.random_forest import RandomForest
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autosklearn.pipeline.components.classification.sgd import SGD
from autosklearn.pipeline.components.classification.passive_aggressive import PassiveAggressive
from autosklearn.pipeline.components.feature_preprocessing\
    .extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor


class BalancingComponentTest(unittest.TestCase):
    def test_balancing_get_weights_treed_single_label(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        self.assertAlmostEqual(
            np.mean(fit_params['classifier:sample_weight']), 1,
        )
        np.testing.assert_allclose(
            fit_params['classifier:sample_weight'],
            np.array([0.625] * 80 + [2.5] * 20),
        )

    def test_balancing_get_weights_treed_multilabel(self):
        Y = np.array([[0, 0, 0]] * 100 + [[1, 0, 0]] * 100 + [[0, 1, 0]] * 100 +
                     [[1, 1, 0]] * 100 + [[0, 0, 1]] * 100 + [[1, 0, 1]] * 10)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        print(fit_params['classifier:sample_weight'])
        self.assertAlmostEqual(
            np.mean(fit_params['classifier:sample_weight']), 1,
        )
        np.testing.assert_allclose(
            fit_params['classifier:sample_weight'],
            np.array([0.85] * 500 + [8.5] * 10),
        )

    def test_balancing_get_weights_svm_sgd(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'libsvm_svc', None, None, None)
        self.assertEqual(("classifier:class_weight", "balanced"),
                         list(init_params.items())[0])
        init_params, fit_params = balancing.get_weights(
            Y, None, 'liblinear_svc_preprocessor', None, None)
        self.assertEqual(("preprocessor:class_weight", "balanced"),
                         list(init_params.items())[0])

    def test_weighting_effect(self):
        data = sklearn.datasets.make_classification(
            n_samples=200, n_features=10, n_redundant=2, n_informative=2,
            n_repeated=2, n_clusters_per_class=2, weights=[0.8, 0.2],
            random_state=1)

        for name, clf, acc_no_weighting, acc_weighting, places in \
                [('adaboost', AdaboostClassifier, 0.810, 0.735, 3),
                 ('decision_tree', DecisionTree, 0.780, 0.643, 3),
                 ('extra_trees', ExtraTreesClassifier, 0.780, 0.8, 3),
                 ('gradient_boosting', GradientBoostingClassifier,
                  0.737, 0.684, 3),
                 ('random_forest', RandomForest, 0.780, 0.789, 3),
                 ('libsvm_svc', LibSVM_SVC, 0.769, 0.72, 3),
                 ('liblinear_svc', LibLinear_SVC, 0.762, 0.735, 3),
                 ('passive_aggressive', PassiveAggressive, 0.642, 0.444, 3),
                 ('sgd', SGD, 0.818, 0.567, 2)
                ]:
            for strategy, acc in [
                ('none', acc_no_weighting),
                ('weighting', acc_weighting)
            ]:
                # Fit
                data_ = copy.copy(data)
                X_train = data_[0][:100]
                Y_train = data_[1][:100]
                X_test = data_[0][100:]
                Y_test = data_[1][100:]

                include = {'classifier': [name],
                           'preprocessor': ['no_preprocessing']}
                classifier = SimpleClassificationPipeline(
                    random_state=1, include=include)
                cs = classifier.get_hyperparameter_search_space()
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(
                    default, random_state=1, include=include)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(
                    sklearn.metrics.f1_score(predictions, Y_test), acc,
                    places=places, msg=(name, strategy))

                # fit_transformer and fit_estimator
                data_ = copy.copy(data)
                X_train = data_[0][:100]
                Y_train = data_[1][:100]
                X_test = data_[0][100:]
                Y_test = data_[1][100:]

                classifier = SimpleClassificationPipeline(
                    default, random_state=1, include=include)
                classifier.set_hyperparameters(configuration=default)
                Xt, fit_params = classifier.fit_transformer(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, **fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(
                    sklearn.metrics.f1_score(predictions, Y_test), acc,
                    places=places)

        for name, pre, acc_no_weighting, acc_weighting in \
                [('extra_trees_preproc_for_classification',
                    ExtraTreesPreprocessorClassification, 0.810, 0.563),
                 ('liblinear_svc_preprocessor', LibLinear_Preprocessor,
                    0.837, 0.576)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                data_ = copy.copy(data)
                X_train = data_[0][:100]
                Y_train = data_[1][:100]
                X_test = data_[0][100:]
                Y_test = data_[1][100:]

                include = {'classifier': ['sgd'], 'preprocessor': [name]}

                classifier = SimpleClassificationPipeline(
                    random_state=1, include=include)
                cs = classifier.get_hyperparameter_search_space()
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier.set_hyperparameters(default)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(
                    sklearn.metrics.f1_score(predictions, Y_test), acc,
                    places=3, msg=(name, strategy))

                # fit_transformer and fit_estimator
                data_ = copy.copy(data)
                X_train = data_[0][:100]
                Y_train = data_[1][:100]
                X_test = data_[0][100:]
                Y_test = data_[1][100:]

                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(
                    default, random_state=1, include=include)
                Xt, fit_params = classifier.fit_transformer(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, **fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(
                    sklearn.metrics.f1_score(predictions, Y_test), acc,
                    places=3)
