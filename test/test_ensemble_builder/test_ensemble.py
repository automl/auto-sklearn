import os
import logging
import shutil
import sys
import time
import unittest
import unittest.mock

import numpy as np

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)

from autosklearn.ensemble_builder import EnsembleBuilder
from autosklearn.metrics import roc_auc


class BackendMock(object):
    
    def __init__(self):
        self.temporary_directory = "test/test_ensemble_builder/data/"
    
    def load_targets_ensemble(self):
        with open(os.path.join(self.temporary_directory,".auto-sklearn","predictions_ensemble","predictions_ensemble_true.npy"),"rb") as fp:
            y = np.load(fp)
        return y


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self.backend = BackendMock()

    def tearDown(self):
        pass

    def testRead(self):
        
        ensbuilder = EnsembleBuilder(backend=self.backend, 
                                    dataset_name="TEST",
                                    task_type=1,  #Binary Classification
                                    metric=roc_auc,
                                    limit=-1, # not used,
                                    seed=0, # important to find the test files
                                    )
        
        success = ensbuilder.read_ensemble_preds()
        self.assertTrue(success, str(ensbuilder.read_preds))
        self.assertEqual(len(ensbuilder.read_preds), 2)
        
        self.assertEqual(ensbuilder.read_preds[
            "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"]
            ["ens_score"], 0.5)
        self.assertEqual(ensbuilder.read_preds[
            "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"]
            ["ens_score"], 1.0)       
                    
    def testNBest(self):
        
        ensbuilder = EnsembleBuilder(backend=self.backend, 
                                    dataset_name="TEST",
                                    task_type=1,  #Binary Classification
                                    metric=roc_auc,
                                    limit=-1, # not used,
                                    seed=0, # important to find the test files
                                    ensemble_nbest=1
                                    )
        
        ensbuilder.read_ensemble_preds()
        sel_keys = ensbuilder.get_n_best_preds()

        self.assertEquals(len(sel_keys), 1)
        self.assertEquals(sel_keys[0],"test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy")
        
    def testFallBackNBest(self):
        
        ensbuilder = EnsembleBuilder(backend=self.backend, 
                                    dataset_name="TEST",
                                    task_type=1,  #Binary Classification
                                    metric=roc_auc,
                                    limit=-1, # not used,
                                    seed=0, # important to find the test files
                                    ensemble_nbest=1
                                    )
        
        ensbuilder.read_ensemble_preds()
        
        ensbuilder.read_preds[
            "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"]["ens_score"] = -1
       
        ensbuilder.read_preds[
            "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"]["ens_score"] = -1
        
        sel_keys = ensbuilder.get_n_best_preds()
        
        self.assertEquals(sel_keys[0],"test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy")
        
    def testGetValidTestPreds(self):
        
        ensbuilder = EnsembleBuilder(backend=self.backend, 
                                    dataset_name="TEST",
                                    task_type=1,  #Binary Classification
                                    metric=roc_auc,
                                    limit=-1, # not used,
                                    seed=0, # important to find the test files
                                    ensemble_nbest=1
                                    )
        
        ensbuilder.read_ensemble_preds()
        
        d2 = "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        d1 = "test/test_ensemble_builder/data/.auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"
        
        sel_keys = ensbuilder.get_n_best_preds()
        
        ensbuilder.get_valid_test_preds(selected_keys=sel_keys)
        
        # selected --> read valid and test predictions
        self.assertIsNot(ensbuilder.read_preds[d2]["y_valid"], None)
        self.assertIsNot(ensbuilder.read_preds[d2]["y_test"], None)
        
        # not selected --> should still be None
        self.assertIs(ensbuilder.read_preds[d1]["y_valid"], None)
        self.assertIs(ensbuilder.read_preds[d1]["y_test"], None)
        
        