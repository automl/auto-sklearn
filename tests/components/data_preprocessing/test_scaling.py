import unittest

import numpy as np
import sklearn.datasets

from ParamSklearn.components.data_preprocessing.rescaling import RescalingChoice
from ParamSklearn.util import get_dataset


class ScalingComponentTest(unittest.TestCase):
    def _test_helper(self, Preprocessor, dataset=None, make_sparse=False):
        X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                          make_sparse=make_sparse)
        original_X_train = X_train.copy()
        configuration_space = Preprocessor.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()

        preprocessor = Preprocessor(random_state=1,
                                    **{hp_name: default[hp_name] for hp_name in
                                       default if default[hp_name] is not None})
        preprocessor = preprocessor.choice
        transformer = preprocessor.fit(X_train, Y_train)
        return transformer.transform(X_train), original_X_train

    def test_boston_is_not_scaled(self):
        data = sklearn.datasets.load_boston()['data']
        self.assertGreaterEqual(np.max(data), 100)

    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            transformation, original = self._test_helper(RescalingChoice,
                                                         dataset='boston')
            # The maximum is around 1.95 for the transformed array...
            self.assertLessEqual(np.max(transformation), 2)
            self.assertFalse((original == transformation).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    def test_default_configuration_with_sparse_data(self):
        preprocessing = self._test_helper(RescalingChoice, dataset='boston',
                                          make_sparse=True)
        transformation, original = preprocessing
        self.assertEqual(original.getnnz(), transformation.getnnz())
        self.assertAlmostEqual(1, transformation.max(), places=6)
        self.assertTrue(~np.allclose(original.data, transformation.data))

    @unittest.skip("Does not work at the moment.")
    def test_preprocessing_dtype(self):
        super(ScalingComponentTest, self)._test_helper(
            RescalingChoice)
