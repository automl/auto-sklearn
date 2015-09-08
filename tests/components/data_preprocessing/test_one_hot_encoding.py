import os
import unittest

import numpy as np
from scipy import sparse

from ParamSklearn.components.data_preprocessing.one_hot_encoding import OneHotEncoder


class OneHotEncoderTest(unittest.TestCase):
    def setUp(self):
        self.categorical = [True,
                            True,
                            True,
                            False,
                            False,
                            True,
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True]
        this_directory = os.path.dirname(__file__)
        self.X_train = np.loadtxt(os.path.join(this_directory, "dataset.pkl"))

    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(random_state=1,
                                         init_params=self.categorical,
                                        **{hp_name: default[hp_name] for hp_name in
                                           default if default[hp_name] is not None})

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                self.assertFalse(
                    (transformations[-1].todense() != transformations[-2].todense()).all())

    def test_default_configuration_sparse_data(self):
        transformations = []

        self.X_train[~np.isfinite(self.X_train)] = 0
        self.X_train = sparse.csc_matrix(self.X_train)

        for i in range(10):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(random_state=1,
                                         init_params=self.categorical,
                                         **{hp_name: default[hp_name] for
                                            hp_name in
                                            default if
                                            default[hp_name] is not None})

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                self.assertFalse(
                    (transformations[-1].todense() != transformations[
                        -2].todense()).all())
