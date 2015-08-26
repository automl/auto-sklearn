import unittest

from ParamSklearn.components.classification.proj_logit import ProjLogitCLassifier
from ParamSklearn.components.feature_preprocessing.gem import GEM
from ParamSklearn.util import _test_preprocessing, PreprocessingTestCase, get_dataset
import sklearn.metrics


class GEMComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(GEM)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_classify(self):
        for i in range(3):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=False)
            configuration_space = GEM.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = GEM(random_state=1,
                               **{hp_name: default[hp_name] for hp_name in
                                  default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = ProjLogitCLassifier(max_epochs = 5, random_state=1)
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.accuracy_score(predictions, Y_test)
            self.assertGreaterEqual(accuracy, 0.94)

    def test_preprocessing_dtype(self):
        super(GEMComponentTest, self)._test_preprocessing_dtype(GEM,
                                                                test_sparse=False)
