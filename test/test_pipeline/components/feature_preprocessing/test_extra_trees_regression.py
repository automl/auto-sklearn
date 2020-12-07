from sklearn.ensemble import ExtraTreesRegressor
from autosklearn.pipeline.components.feature_preprocessing.\
    extra_trees_preproc_for_regression import \
    ExtraTreesPreprocessorRegression
from autosklearn.pipeline.util import _test_preprocessing, \
    PreprocessingTestCase, get_dataset
import sklearn.metrics


class ExtraTreesRegressionComponentTest(PreprocessingTestCase):
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
            error = sklearn.metrics.mean_squared_error(predictions, Y_test)
            self.assertAlmostEqual(error, 18.074952764044944, places=2)

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
            error = sklearn.metrics.mean_squared_error(predictions, Y_test)
            self.assertAlmostEqual(error, 55.69613978965742, places=2)

    def test_preprocessing_dtype(self):
        super(ExtraTreesRegressionComponentTest, self).\
            _test_preprocessing_dtype(ExtraTreesPreprocessorRegression)
