import unittest

from AutoSklearn.components.regression.ridge_regression import RidgeRegression
from AutoSklearn.components.preprocessing.kitchen_sinks import RandomKitchenSinks
from AutoSklearn.util import _test_regressor, get_dataset

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        configuration_space = RidgeRegression.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        configuration_space_preproc = RandomKitchenSinks.get_hyperparameter_search_space()
        default_preproc = configuration_space_preproc.get_default_configuration()

        for i in range(10):
            # This should be a bad results
            predictions, targets = _test_regressor(RidgeRegression,
                                                   dataset='diabetes')
            self.assertAlmostEqual(-3.726787582018825,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

            # This should be much more better
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes',
                                                           make_sparse=False)
            preprocessor = RandomKitchenSinks(
                random_state=1,
                **{hp.hyperparameter.name: hp.value for hp in default_preproc.values.values()})

            transformer = preprocessor.fit(X_train, Y_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            regressor = RidgeRegression(
                random_state=1,
                **{hp.hyperparameter.name: hp.value for hp in default.values.values()})
            predictor = regressor.fit(X_train_transformed, Y_train)
            predictions = predictor.predict(X_test_transformed)

            self.assertAlmostEqual(0.32731125809612438, #0.24658871483206091
                sklearn.metrics.r2_score(y_true=Y_test, y_pred=predictions))