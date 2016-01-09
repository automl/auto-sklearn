__author__ = 'feurerm'

import copy
import unittest

import numpy as np
import sklearn.datasets
import sklearn.metrics

from autosklearn.pipeline.components.data_preprocessing.balancing import Balancing
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier
from autosklearn.pipeline.components.classification.gradient_boosting import GradientBoostingClassifier
from autosklearn.pipeline.components.classification.random_forest import RandomForest
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autosklearn.pipeline.components.classification.sgd import SGD
from autosklearn.pipeline.components.feature_preprocessing\
    .extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor


class BalancingComponentTest(unittest.TestCase):
    def test_balancing_get_weights_treed_single_label(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        self.assertTrue(np.allclose(fit_params['classifier:sample_weight'],
                                    np.array([0.4] * 80 + [1.6] * 20)))
        #init_params, fit_params = balancing.get_weights(
        #    Y, None, 'extra_trees_preproc_for_classification', None, None)
        #self.assertTrue(np.allclose(fit_params['preprocessor:sample_weight'],
        #                            np.array([0.4] * 80 + [1.6] * 20)))

    def test_balancing_get_weights_treed_multilabel(self):
        Y = np.array([[0, 0, 0]] * 100 + [[1, 0, 0]] * 100 + [[0, 1, 0]] * 100 +
                     [[1, 1, 0]] * 100 + [[0, 0, 1]] * 100 + [[1, 0, 1]] * 10)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        self.assertTrue(np.allclose(fit_params['classifier:sample_weight'],
                                    np.array([0.4] * 500 + [4.0] * 10)))
        #init_params, fit_params = balancing.get_weights(
        #    Y, None, 'extra_trees_preproc_for_classification', None, None)
        #self.assertTrue(np.allclose(fit_params['preprocessor:sample_weight'],
        #                            np.array([0.4] * 500 + [4.0] * 10)))

    def test_balancing_get_weights_svm_sgd(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'libsvm_svc', None, None, None)
        self.assertEqual(("classifier:class_weight", "auto"),
                         list(init_params.items())[0])
        init_params, fit_params = balancing.get_weights(
            Y, None, 'liblinear_svc_preprocessor', None, None)
        self.assertEqual(("preprocessor:class_weight", "auto"),
                         list(init_params.items())[0])

    def test_weighting_effect(self):
        data = sklearn.datasets.make_classification(
            n_samples=1000, n_features=20, n_redundant=5, n_informative=5,
            n_repeated=2, n_clusters_per_class=2, weights=[0.8, 0.2],
            random_state=1)

        for name, clf, acc_no_weighting, acc_weighting in \
                [('adaboost', AdaboostClassifier, 0.709, 0.662),
                 ('decision_tree', DecisionTree, 0.683, 0.726),
                 ('extra_trees', ExtraTreesClassifier, 0.812, 0.812),
                 ('gradient_boosting', GradientBoostingClassifier,
                    0.800, 0.760),
                 ('random_forest', RandomForest, 0.849, 0.780),
                 ('libsvm_svc', LibSVM_SVC, 0.571, 0.658),
                 ('liblinear_svc', LibLinear_SVC, 0.685, 0.699),
                 ('sgd', SGD, 0.602, 0.720)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                # Fit
                data_ = copy.copy(data)
                X_train = data_[0][:700]
                Y_train = data_[1][:700]
                X_test = data_[0][700:]
                Y_test = data_[1][700:]

                cs = SimpleClassificationPipeline.\
                    get_hyperparameter_search_space(
                        include={'classifier': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                    sklearn.metrics.f1_score(predictions, Y_test),
                    places=3)

                # pre_transform and fit_estimator
                data_ = copy.copy(data)
                X_train = data_[0][:700]
                Y_train = data_[1][:700]
                X_test = data_[0][700:]
                Y_test = data_[1][700:]

                cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
                    include={'classifier': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(default, random_state=1)
                Xt, fit_params = classifier.pre_transform(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, fit_params=fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.f1_score(
                                           predictions, Y_test),
                                       places=3)

        for name, pre, acc_no_weighting, acc_weighting in \
                [('extra_trees_preproc_for_classification',
                    ExtraTreesPreprocessorClassification, 0.685, 0.589),
                 ('liblinear_svc_preprocessor', LibLinear_Preprocessor,
                    0.714, 0.596)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                data_ = copy.copy(data)
                X_train = data_[0][:700]
                Y_train = data_[1][:700]
                X_test = data_[0][700:]
                Y_test = data_[1][700:]

                cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
                    include={'classifier': ['sgd'], 'preprocessor': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.f1_score(
                                           predictions, Y_test),
                                       places=3)

                # pre_transform and fit_estimator
                data_ = copy.copy(data)
                X_train = data_[0][:700]
                Y_train = data_[1][:700]
                X_test = data_[0][700:]
                Y_test = data_[1][700:]

                cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
                    include={'classifier': ['sgd'], 'preprocessor': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = SimpleClassificationPipeline(default, random_state=1)
                Xt, fit_params = classifier.pre_transform(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, fit_params=fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.f1_score(
                                           predictions, Y_test),
                                       places=3)