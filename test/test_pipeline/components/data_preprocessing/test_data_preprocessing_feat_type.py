from autosklearn.pipeline.components.data_preprocessing.feature_type import (
    FeatTypeSplit,
)

import unittest


class PreprocessingPipelineFeatTypeTest(unittest.TestCase):

    num_numerical = 6
    num_categorical = 3
    num_text = 11

    def test_single_type(self):
        DPP = FeatTypeSplit(feat_type={"A": "numerical"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "numerical"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("text", key.split(":")[0])
            self.assertNotIn("categorical", key.split(":")[0])
        self.assertEqual(len(cs), self.num_numerical)

        DPP = FeatTypeSplit(feat_type={"A": "categorical"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "categorical"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("text", key.split(":")[0])
            self.assertNotIn("numerical", key.split(":")[0])
        self.assertEqual(len(cs), self.num_categorical)

        DPP = FeatTypeSplit(feat_type={"A": "string"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "string"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("numerical", key.split(":")[0])
            self.assertNotIn("categorical", key.split(":")[0])
        self.assertEqual(len(cs), self.num_text)

    def test_dual_type(self):
        DPP = FeatTypeSplit(feat_type={"A": "numerical", "B": "categorical"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "numerical", "B": "categorical"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("text", key.split(":")[0])
        self.assertEqual(len(cs), self.num_numerical + self.num_categorical)

        DPP = FeatTypeSplit(feat_type={"A": "categorical", "B": "string"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "categorical", "B": "string"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("numerical", key.split(":")[0])
        self.assertEqual(len(cs), self.num_categorical + self.num_text)

        DPP = FeatTypeSplit(feat_type={"A": "string", "B": "numerical"})
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "string", "B": "numerical"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        for key in cs.get_hyperparameters_dict().keys():
            self.assertNotIn("categorical", key.split(":")[0])
        self.assertEqual(len(cs), self.num_text + self.num_numerical)

    def test_triple_type(self):
        DPP = FeatTypeSplit(
            feat_type={"A": "numerical", "B": "categorical", "C": "string"}
        )
        cs = DPP.get_hyperparameter_search_space(
            feat_type={"A": "numerical", "B": "categorical", "C": "string"},
            dataset_properties={
                "task": 1,
                "sparse": False,
                "multilabel": False,
                "multiclass": False,
                "target_type": "classification",
                "signed": False,
            },
        )
        truth_table = [False] * 3
        for key in cs.get_hyperparameters_dict().keys():
            if "text" in key.split(":")[0]:
                truth_table[0] = True
            elif "categorical" in key.split(":")[0]:
                truth_table[1] = True
            elif "numerical" in key.split(":")[0]:
                truth_table[2] = True

        self.assertEqual(sum(truth_table), 3)
        self.assertEqual(
            len(cs), self.num_numerical + self.num_categorical + self.num_text
        )
