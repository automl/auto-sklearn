import unittest

import HPOlibConfigSpace.configuration_space

import ParamSklearn.base
import ParamSklearn.components.preprocessing
import ParamSklearn.components.classification

class BaseTest(unittest.TestCase):
    def test_get_hyperparameter_configuration_space_3choices(self):
        base = ParamSklearn.base.ParamSklearnBaseEstimator

        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {}
        exclude = {}
        include = {}
        pipeline = [('p0', ParamSklearn.components.preprocessing._preprocessors[
                        'preprocessor']),
                    ('p1', ParamSklearn.components.preprocessing._preprocessors[
                        'preprocessor']),
                    ('c', ParamSklearn.components.classification._classifiers[
                        'classifier'])]
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices), 14)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices), 16)
        self.assertEqual(143, len(cs.forbidden_clauses))

        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'sparse': True}
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         11)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         16)
        self.assertEqual(387, len(cs.forbidden_clauses))
