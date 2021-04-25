import unittest
import unittest.mock

import ConfigSpace.configuration_space

import autosklearn.pipeline.base
import autosklearn.pipeline.components.base
import autosklearn.pipeline.components.feature_preprocessing
import autosklearn.pipeline.components.classification


class BasePipelineMock(autosklearn.pipeline.base.BasePipeline):
    def __init__(self):
        pass


class BaseTest(unittest.TestCase):
    def test_get_hyperparameter_configuration_space_3choices(self):
        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification'}
        exclude = {}
        include = {}
        pipeline = [('p0',
                     autosklearn.pipeline.components.feature_preprocessing
                     .FeaturePreprocessorChoice(dataset_properties)),
                    ('p1',
                     autosklearn.pipeline.components.feature_preprocessing
                     .FeaturePreprocessorChoice(dataset_properties)),
                    ('c', autosklearn.pipeline.components.classification
                     .ClassifierChoice(dataset_properties))]

        base = BasePipelineMock()
        cs = base._get_base_search_space(cs, dataset_properties,
                                         exclude, include, pipeline)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         13)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         15)

        # for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #     print(clause)
        self.assertEqual(148, len(cs.forbidden_clauses))

        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'signed': True}
        include = {'c': ['multinomial_nb']}
        cs = base._get_base_search_space(cs, dataset_properties,
                                         exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         13)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         10)
        self.assertEqual(len(cs.get_hyperparameter("c:__choice__").choices),
                         1)
        # Mostly combinations of p0 making the data unsigned and p1 not
        # changing the values of the data points
        # for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)
        self.assertEqual(64, len(cs.forbidden_clauses))

        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'signed': True}
        include = {}
        cs = base._get_base_search_space(cs, dataset_properties,
                                         exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         13)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         15)
        self.assertEqual(len(cs.get_hyperparameter("c:__choice__").choices),
                         16)
        # for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)
        self.assertEqual(110, len(cs.forbidden_clauses))

        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification', 'sparse': True}
        cs = base._get_base_search_space(cs, dataset_properties,
                                         exclude, include, pipeline)
        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         12)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         15)
        # for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)
        self.assertEqual(419, len(cs.forbidden_clauses))

        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        dataset_properties = {'target_type': 'classification',
                              'sparse': True, 'signed': True}
        cs = base._get_base_search_space(cs, dataset_properties,
                                         exclude, include, pipeline)

        self.assertEqual(len(cs.get_hyperparameter("p0:__choice__").choices),
                         12)
        self.assertEqual(len(cs.get_hyperparameter("p1:__choice__").choices),
                         15)
        # Data is guaranteed to be positive in cases like densifier,
        # extra_trees_preproc, multinomial_nb -> less constraints
        # for clause in sorted([str(clause) for clause in cs.forbidden_clauses]):
        #    print(clause)
        self.assertEqual(359, len(cs.forbidden_clauses))

    def test_init_params_handling(self):
        """
        Makes sure that init params is properly passed to nodes

        Also, makes sure that _check_init_params_honored raises the expected exceptions
        """
        cs = ConfigSpace.configuration_space.ConfigurationSpace()
        base = BasePipelineMock()
        base.dataset_properties = {}

        # Make sure that component irrespective, we check the init params
        for node_type in [
            autosklearn.pipeline.components.base.AutoSklearnComponent,
            autosklearn.pipeline.components.base.AutoSklearnChoice,
            autosklearn.pipeline.base.BasePipeline,
        ]:

            # We have couple of posibilities
            for init_params, expected_init_params in [
                ({}, {}),
                (None, None),
                ({'M:key': 'value'}, {'key': 'value'}),
            ]:
                node = unittest.mock.Mock(
                    spec=autosklearn.pipeline.components.base.AutoSklearnComponent
                )
                node.get_hyperparameter_search_space.return_value = cs
                node.key = 'value'
                base.steps = [('M', node)]
                base.set_hyperparameters(cs.sample_configuration(), init_params=init_params)
                self.assertEqual(node.set_hyperparameters.call_args[1]['init_params'],
                                 expected_init_params)

            # Check for proper exception raising
            node = unittest.mock.Mock(
                spec=autosklearn.pipeline.components.base.AutoSklearnComponent
            )
            node.get_hyperparameter_search_space.return_value = cs
            base.steps = [('M', node)]
            with self.assertRaisesRegex(ValueError, "Unsupported argument to init_params"):
                base.set_hyperparameters(cs.sample_configuration(), init_params={'key': 'value'})

            # An invalid node name is passed
            with self.assertRaisesRegex(ValueError, "The current node name specified via key"):
                base.set_hyperparameters(cs.sample_configuration(), init_params={'N:key': 'value'})

            # The value was not properly set -- Here it happens because the
            # object is a magic mock, calling the method doesn't set a new parameter
            with self.assertRaisesRegex(ValueError, "Cannot properly set the pair"):
                base.set_hyperparameters(cs.sample_configuration(), init_params={'M:key': 'value'})

    def test_include_exclude_validation(self):
        """
        Makes sure include and exclude arguments are validated and raises expected exception
        on error
        """
        base = BasePipelineMock()
        dataset_properties = {'target_type': 'classification'}
        base.dataset_properties = dataset_properties
        base.steps = [('p0',
                       autosklearn.pipeline.components.feature_preprocessing
                       .FeaturePreprocessorChoice(dataset_properties)),
                      ('p1',
                       autosklearn.pipeline.components.feature_preprocessing
                       .FeaturePreprocessorChoice(dataset_properties)),
                      ('c', autosklearn.pipeline.components.classification
                       .ClassifierChoice(dataset_properties))]

        def assert_value_error(include=None, exclude=None):
            base.include = include
            base.exclude = exclude
            with self.assertRaises(ValueError):
                base._validate_include_exclude_params()

        # Same key in include and exclude argument
        assert_value_error(include={'c': ['adaboost']}, exclude={'c': ['sgd']})

        # Invalid key in the exclude argument
        assert_value_error(exclude={'p2': ['pca']})

        # Invalid value type for the key in the include argument
        assert_value_error(include={'c': ('adaboost', 'sgd')}, exclude=None)

        # Empty list of the key in the include argument
        assert_value_error(include={'c': []})

        # Invalid component in the list value for the key in the include argument
        assert_value_error(include={'c': ['pca']})

        # Case when all conditions passed for include and exclude
        base.include = {'c': ['adaboost', 'sgd']}
        base.exclude = {'p1': ['pca']}
        self.assertIsNone(base._validate_include_exclude_params())
