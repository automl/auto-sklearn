import unittest

import HPOlibConfigSpace.configuration_space

import ParamSklearn.base
import ParamSklearn.components.feature_preprocessing
import ParamSklearn.components.classification

class BaseTest(unittest.TestCase):
    def test_get_hyperparameter_configuration_space_3choices(self):
        base = ParamSklearn.base.ParamSklearnBaseEstimator

        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification'}
        exclude = {}
        include = {}
        pipeline = [('p0', ParamSklearn.components.feature_preprocessing._preprocessors[
                        'preprocessor']),
                    ('p1', ParamSklearn.components.feature_preprocessing._preprocessors[
                        'preprocessor']),
                    ('c', ParamSklearn.components.classification._classifiers[
                        'classifier'])]
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices), 14)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices), 16)

        #for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print clause
        self.assertEqual(151, len(cs.forbidden_clauses))

        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'signed': True}
        include = {'c': ['multinomial_nb']}
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         14)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         10)
        self.assertEqual(len(cs.get_hyperparameter("c:__choice__").choices),
                         1)
        # Mostly combinations of p0 making the data unsigned and p1 not
        # changing the values of the data points
        self.assertEqual(74, len(cs.forbidden_clauses))


        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'signed': True}
        include = {}
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         14)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         16)
        self.assertEqual(len(cs.get_hyperparameter("c:__choice__").choices),
                         16)
        self.assertEqual(126, len(cs.forbidden_clauses))


        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'sparse': True}
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         11)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         16)
        self.assertEqual(409, len(cs.forbidden_clauses))
        #for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)

        cs = HPOlibConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification',
                              'sparse': True, 'signed': True}
        cs = base._get_hyperparameter_search_space(cs, dataset_properties,
                                                   exclude, include, pipeline)

        #for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         11)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         16)
        # Data is guaranteed to be positive in cases like densifier,
        # extra_trees_preproc, multinomial_nb -> less constraints
        self.assertEqual(364, len(cs.forbidden_clauses))


