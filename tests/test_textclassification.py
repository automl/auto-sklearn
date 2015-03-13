import unittest

from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from ParamSklearn.textclassification import ParamSklearnTextClassifier


class TextClassificationTest(unittest.TestCase):
    def test_get_hyperparameter_search_space(self):
        cs = ParamSklearnTextClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(89, len(hyperparameters))
        # The three parameters which are always active are classifier,
        # preprocessor and imputation strategy
        self.assertEqual(len(hyperparameters) - 3, len(conditions))
        self.assertNotIn("rescaling", cs.get_hyperparameter(
            "preprocessor").choices)
        self.assertRaisesRegexp(KeyError, "Hyperparameter "
                                          "'rescaling:strategy' does not "
                                          "exist in this configuration "
                                          "space.", cs.get_hyperparameter,
                                "rescaling:strategy")