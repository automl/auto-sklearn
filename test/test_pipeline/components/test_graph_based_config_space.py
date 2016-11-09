import unittest

from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnComponent, LeafNodeConfigSpaceBuilder
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.config_space import InvalidDataArtifactsException
from autosklearn.pipeline.components.data_preprocessing.imputation.imputation import Imputation
from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding import OneHotEncoder
from autosklearn.pipeline.components.data_preprocessing.rescaling import \
    RescalingChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.parallel import ParallelAutoSklearnComponent
from autosklearn.pipeline.components.regression import RegressorChoice
from autosklearn.pipeline.components.serial import SerialAutoSklearnComponent


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
        pipeline = SerialAutoSklearnComponent(
            [ParallelAutoSklearnComponent(
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
        pipeline = SerialAutoSklearnComponent(
            [
                RescalingChoice(),
                ClassifierChoice()
            ]
        )

        cs = pipeline.get_config_space()
        print(cs)

    def test_simple_classification_pipeline(self):
        pipeline = SerialAutoSklearnComponent(
            [
                ("one_hot_encoding", OneHotEncoder()),
                ("imputation", Imputation()),
                ("rescaling", RescalingChoice()),
                ("preprocessor", FeaturePreprocessorChoice()),
                ("classifier", ClassifierChoice())
            ]
        )

        cs = pipeline.get_config_space()
        print(cs)

    def test_set_simple_classification_pipeline(self):
        pipeline = SerialAutoSklearnComponent(
            [
                ("one_hot_encoding", OneHotEncoder()),
                ("imputation", Imputation()),
                ("rescaling", RescalingChoice()),
                ("preprocessor", FeaturePreprocessorChoice()),
                ("classifier", ClassifierChoice())
            ]
        )

        cs = pipeline.get_config_space()
        default = cs.get_default_configuration()
        pipeline.set_hyperparameters(default)

    def test_simple_regression_pipeline(self):
        pipeline = SerialAutoSklearnComponent(
            [
                ("one_hot_encoding", OneHotEncoder()),
                ("imputation", Imputation()),
                ("rescaling", RescalingChoice()),
                ("preprocessor", FeaturePreprocessorChoice()),
                ("regressor", RegressorChoice())
            ]
        )

        cs = pipeline.get_config_space()
        default = cs.get_default_configuration()
        pipeline.set_hyperparameters(default)

    def test_two_parallel_data_processing_pipelines(self):
        pipeline = SerialAutoSklearnComponent(
            [
                ParallelAutoSklearnComponent([
                    SerialAutoSklearnComponent(
                        [
                            ("one_hot_encoding", OneHotEncoder()),
                            ("imputation", Imputation()),
                            ("rescaling", RescalingChoice()),
                            ("preprocessor", FeaturePreprocessorChoice())
                        ]
                    ),
                    SerialAutoSklearnComponent(
                        [
                            ("one_hot_encoding", OneHotEncoder()),
                            ("imputation", Imputation()),
                            ("rescaling", RescalingChoice()),
                            ("preprocessor", FeaturePreprocessorChoice())
                        ]
                    )
                ]),
                ("classifier", ClassifierChoice())
            ]
        )

        cs = pipeline.get_config_space()
        default = cs.get_default_configuration()
        pipeline.set_hyperparameters(default)
