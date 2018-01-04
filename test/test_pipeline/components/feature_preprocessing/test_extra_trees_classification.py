import unittest

from sklearn.linear_model import RidgeClassifier
from autosklearn.pipeline.components.feature_preprocessing.\
    extra_trees_preproc_for_classification import \
    ExtraTreesPreprocessorClassification
from autosklearn.pipeline.util import _test_preprocessing, \
    PreprocessingTestCase, get_dataset
import sklearn.metrics


class ExtreTreesClassificationComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(
                ExtraTreesPreprocessorClassification)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_classify(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=False)
            configuration_space = ExtraTreesPreprocessorClassification.\
                get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = ExtraTreesPreprocessorClassification(
                    random_state=1,
                    **{hp_name: default[hp_name] for hp_name in default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = RidgeClassifier()
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.accuracy_score(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 0.87310261080752882, places=2)

    def test_default_configuration_classify_sparse(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=True)
            configuration_space = ExtraTreesPreprocessorClassification.\
                get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = ExtraTreesPreprocessorClassification(
                    random_state=1,
                    **{hp_name: default[hp_name] for hp_name in default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = RidgeClassifier()
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.accuracy_score(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 0.43715846994535518, places=2)

    def test_preprocessing_dtype(self):
        super(ExtreTreesClassificationComponentTest, self).\
            _test_preprocessing_dtype(ExtraTreesPreprocessorClassification)
