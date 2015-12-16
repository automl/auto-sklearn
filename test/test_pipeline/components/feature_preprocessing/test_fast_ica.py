import unittest

from sklearn.linear_model import Ridge
from autosklearn.pipeline.components.feature_preprocessing.fast_ica import \
    FastICA
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase, \
    get_dataset
import sklearn.metrics


class FastICAComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(FastICA,
                                                       dataset="diabetes")
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_regression(self):
        for i in range(5):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes')
            configuration_space = FastICA.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = FastICA(random_state=1,
                                   **{hp_name: default[hp_name] for hp_name in
                                      default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = Ridge()
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.r2_score(Y_test, predictions)
            self.assertAlmostEqual(accuracy, 0.32614416980439365)

    @unittest.skip("Always returns float64")
    def test_preprocessing_dtype(self):
        super(FastICAComponentTest,
              self)._test_preprocessing_dtype(FastICA, dataset='diabetes')

