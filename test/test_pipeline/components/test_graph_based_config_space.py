import unittest

from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnComponent
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import \
    Balancing
from autosklearn.pipeline.components.data_preprocessing.rescaling import \
    RescalingChoice
from autosklearn.pipeline.graph_based_config_space import LeafNodeConfigSpaceBuilder, InvalidDataArtifactsException
from autosklearn.pipeline.serial_flow_component import SerialFlow, ParallelFlow




class BadStub(AutoSklearnComponent):

    def get_config_space(self):
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

    def get_config_space(self):
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