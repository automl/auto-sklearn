import unittest

from ParamSklearn.components.regression.ridge_regression import RidgeRegression
from ParamSklearn.components.feature_preprocessing.kitchen_sinks import RandomKitchenSinks
from ParamSklearn.util import _test_regressor, get_dataset

import sklearn.metrics


class RidgeComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        configuration_space = RidgeRegression.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        configuration_space_preproc = RandomKitchenSinks.get_hyperparameter_search_space()
        default_preproc = configuration_space_preproc.get_default_configuration()

        for i in range(10):
            # This should be a bad results
            predictions, targets = _test_regressor(RidgeRegression,)
            self.assertAlmostEqual(0.32614416980439365,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

            # This should be much more better
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes',
                                                           make_sparse=False)
            preprocessor = RandomKitchenSinks(
                random_state=1,
                **{hp_name: default_preproc[hp_name] for hp_name in
                   default_preproc if default_preproc[hp_name] is not None})

            transformer = preprocessor.fit(X_train, Y_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            regressor = RidgeRegression(
                random_state=1,
                **{hp_name: default[hp_name] for hp_name in
                   default if default[hp_name] is not None})
            predictor = regressor.fit(X_train_transformed, Y_train)
            predictions = predictor.predict(X_test_transformed)

            self.assertAlmostEqual(0.37183512452087852,
                sklearn.metrics.r2_score(y_true=Y_test, y_pred=predictions))