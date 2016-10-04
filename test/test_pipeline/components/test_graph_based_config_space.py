import unittest

from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnComponent
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import \
    Balancing
from autosklearn.pipeline.components.data_preprocessing.imputation.imputation import Imputation
from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding import OneHotEncoder
from autosklearn.pipeline.components.data_preprocessing.rescaling import \
    RescalingChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.graph_based_config_space import LeafNodeConfigSpaceBuilder, InvalidDataArtifactsException
from autosklearn.pipeline.serial_flow_component import SerialFlow, ParallelFlow


class BadStub(AutoSklearnComponent):

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter(name="loss", choices=["linear", "square", "exponential"], default="linear"))
        return cs

    def get_config_space_builder(self):
        cs = LeafNodeConfigSpaceBuilder(self)
        return cs

    def transform_data_description(self, data_description):
        data_description.append('exception')
        return data_description


class Stub(AutoSklearnComponent):

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter(name="loss", choices=["linear", "square", "exponential"], default="linear"))
        return cs

    def get_config_space_builder(self):
        cs = LeafNodeConfigSpaceBuilder(self)
        return cs

    def transform_data_description(self, data_description):
        if 'exception' in data_description:
            raise InvalidDataArtifactsException(['exception'])
        else:
            return data_description



class SimplePipelineConfigSpaceTest(unittest.TestCase):

    def test_lol(self):
        pipeline = SerialFlow(
            [ParallelFlow(
                [BadStub(),
                 Stub(),
                 Stub(),
                 Stub()]
            ),
             ('failinig_stub', Stub())]
        )

        cs = pipeline.get_config_space()
        print(cs)

    def test_pipeline_with_choice_nodes(self):
        pipeline = SerialFlow(
            [
                RescalingChoice(),
                ClassifierChoice()
            ]
        )

        cs = pipeline.get_config_space()
        print(cs)

    def test_simple_classification_pipeline(self):
        pipeline = SerialFlow(
            [
                ("one_hot_encoding", OneHotEncoder()),
                ("imputation", Imputation()),
                ("rescaling", RescalingChoice()),
                ("balancing", Balancing()),
                ("preprocessor", FeaturePreprocessorChoice()),
                ("classifier", ClassifierChoice())
            ]
        )

        cs = pipeline.get_config_space()
        print(cs)

    def test_simple_regression_pipeline(self):
        pass

    def test_two_parallel_data_processing_pipelines(self):
        pass