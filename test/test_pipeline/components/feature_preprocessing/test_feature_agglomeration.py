from sklearn.ensemble import RandomForestClassifier
from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import \
    FeatureAgglomeration
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase, \
    get_dataset
import sklearn.metrics


class FeatureAgglomerationComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(FeatureAgglomeration)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_classify(self):
        for i in range(3):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=False)
            configuration_space = FeatureAgglomeration.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = FeatureAgglomeration(random_state=1,
                                                **{hp_name: default[hp_name] for
                                                   hp_name in default})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = RandomForestClassifier(random_state=1)
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.accuracy_score(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 0.8761384335154827)

    def test_preprocessing_dtype(self):
        super(FeatureAgglomerationComponentTest,
              self)._test_preprocessing_dtype(FeatureAgglomeration,
                                              test_sparse=False)
