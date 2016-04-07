import unittest

from sklearn.ensemble import ExtraTreesRegressor
from autosklearn.pipeline.components.feature_preprocessing.\
    extra_trees_preproc_for_regression import \
    ExtraTreesPreprocessorRegression
from autosklearn.pipeline.util import _test_preprocessing, \
    PreprocessingTestCase, get_dataset
import sklearn.metrics


class ExtreTreesRegressionComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(
                ExtraTreesPreprocessorRegression)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_regression(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston',
                                                           make_sparse=False)
            configuration_space = ExtraTreesPreprocessorRegression.\
                get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = ExtraTreesPreprocessorRegression(
                    random_state=1,
                    **{hp_name: default[hp_name] for hp_name in default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a regressor on top
            regressor = ExtraTreesRegressor(random_state=1)
            predictor = regressor.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.mean_squared_error(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 20.193400000000004, places=2)

    def test_default_configuration_classify_sparse(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston',
                                                           make_sparse=True)
            configuration_space = ExtraTreesPreprocessorRegression.\
                get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = ExtraTreesPreprocessorRegression(
                    random_state=1,
                    **{hp_name: default[hp_name] for hp_name in default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a regressor on top
            regressor = ExtraTreesRegressor(random_state=1)
            predictor = regressor.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.mean_squared_error(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 62.485374939528718, places=2)

    def test_preprocessing_dtype(self):
        super(ExtreTreesRegressionComponentTest, self).\
            _test_preprocessing_dtype(ExtraTreesPreprocessorRegression)