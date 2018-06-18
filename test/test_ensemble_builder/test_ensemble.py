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

from autosklearn.ensemble_builder import EnsembleBuilder, Y_ENSEMBLE, Y_VALID, Y_TEST
from autosklearn.metrics import roc_auc


class BackendMock(object):
    
    def __init__(self):
        this_directory = os.path.abspath(
            os.path.dirname(__file__)
        )
        self.temporary_directory = os.path.join(
            this_directory, 'data',
        )
    
    def load_targets_ensemble(self):
        with open(os.path.join(
            self.temporary_directory,
            ".auto-sklearn",
            "predictions_ensemble",
            "predictions_ensemble_true.npy"
        ),"rb") as fp:
            y = np.load(fp)
        return y
    
class EnsembleBuilderMemMock(EnsembleBuilder):
    
    def fit_ensemble(self,selected_keys):
        np.ones([10000000,1000000])


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self.backend = BackendMock()

    def tearDown(self):
        pass

    def testRead(self):
        
        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  #Binary Classification
            metric=roc_auc,
            limit=-1,  # not used,
            seed=0,  # important to find the test files
        )

        success = ensbuilder.read_ensemble_preds()
        self.assertTrue(success, str(ensbuilder.read_preds))
        self.assertEqual(len(ensbuilder.read_preds), 2)

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"
        )
        self.assertEqual(ensbuilder.read_preds[filename]["ens_score"], 0.5)

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        )
        self.assertEqual(ensbuilder.read_preds[filename]["ens_score"], 1.0)
                    
    def testNBest(self):
        
        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  #Binary Classification
            metric=roc_auc,
            limit=-1, # not used,
            seed=0, # important to find the test files
            ensemble_nbest=1,
        )
        
        ensbuilder.read_ensemble_preds()
        sel_keys = ensbuilder.get_n_best_preds()

        self.assertEquals(len(sel_keys), 1)

        fixture = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        )
        self.assertEquals(sel_keys[0], fixture)
        
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

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        )
        ensbuilder.read_preds[filename]["ens_score"] = -1

        filename = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"
        )
        ensbuilder.read_preds[filename]["ens_score"] = -1
        
        sel_keys = ensbuilder.get_n_best_preds()

        fixture = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"
        )
        self.assertEquals(sel_keys[0], fixture)
        
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
        
        d2 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        )
        d1 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1.npy"
        )
        
        sel_keys = ensbuilder.get_n_best_preds()
        
        ensbuilder.get_valid_test_preds(selected_keys=sel_keys)
        
        # selected --> read valid and test predictions
        self.assertIsNotNone(ensbuilder.read_preds[d2][Y_VALID])
        self.assertIsNotNone(ensbuilder.read_preds[d2][Y_TEST])
        
        # not selected --> should still be None
        self.assertIsNone(ensbuilder.read_preds[d1][Y_VALID])
        self.assertIsNone(ensbuilder.read_preds[d1][Y_TEST])
        
    def testEntireEnsembleBuilder(self):
        
        ensbuilder = EnsembleBuilder(
            backend=self.backend,
            dataset_name="TEST",
            task_type=1,  #Binary Classification
            metric=roc_auc,
            limit=-1, # not used,
            seed=0, # important to find the test files
            ensemble_nbest=2,
        )
        ensbuilder.SAVE2DISC = False
        
        ensbuilder.read_ensemble_preds()

        d2 = os.path.join(
            self.backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2.npy"
        )

        sel_keys = ensbuilder.get_n_best_preds()
        self.assertGreater(len(sel_keys), 0)
        
        ensemble = ensbuilder.fit_ensemble(selected_keys=sel_keys)
        print(ensemble, sel_keys)
        
        n_sel_valid, n_sel_test = ensbuilder.get_valid_test_preds(selected_keys=sel_keys)
        
        # both valid and test prediction files are available
        self.assertGreater(len(n_sel_valid), 0)
        self.assertEqual(n_sel_valid, n_sel_test)

        y_valid = ensbuilder.predict(
            set_="valid",
            ensemble=ensemble,
            selected_keys=n_sel_valid,
            n_preds=len(sel_keys),
            index_run=1,
        )
        y_test = ensbuilder.predict(
            set_="test",
            ensemble=ensemble,
            selected_keys=n_sel_test,
            n_preds=len(sel_keys),
            index_run=1,
        )

        # predictions for valid and test are the same
        # --> should results in the same predictions
        np.testing.assert_array_almost_equal(y_valid, y_test)

        # since d2 provides perfect predictions
        # it should get a higher weight
        # so that y_valid should be exactly y_valid_d2
        y_valid_d2 = ensbuilder.read_preds[d2][Y_VALID][:, 1]
        np.testing.assert_array_almost_equal(y_valid, y_valid_d2)
        
    def testMain(self):
        
        ensbuilder = EnsembleBuilder(backend=self.backend, 
                                    dataset_name="TEST",
                                    task_type=1,  #Binary Classification
                                    metric=roc_auc,
                                    limit=-1, # not used,
                                    seed=0, # important to find the test files
                                    ensemble_nbest=2,
                                    max_iterations=1 # prevents infinite loop
                                    )
        ensbuilder.SAVE2DISC = False
        
        ensbuilder.main()
        
        self.assertEqual(len(ensbuilder.read_preds), 2)
        self.assertIsNotNone(ensbuilder.last_hash)
        self.assertIsNotNone(ensbuilder.y_true_ensemble)
        
    def testLimit(self):
        
                
        ensbuilder = EnsembleBuilderMemMock(backend=self.backend, 
                                            dataset_name="TEST",
                                            task_type=1,  #Binary Classification
                                            metric=roc_auc,
                                            limit=1000, # not used,
                                            seed=0, # important to find the test files
                                            ensemble_nbest=10,
                                            max_iterations=1, # prevents infinite loop
                                            memory_limit=10 # small memory limit to trigger MemoryException
                                            )
        ensbuilder.SAVE2DISC = False
        
        ensbuilder.run()
        
        # it should try to reduce ensemble_nbest until it also failed at 2
        self.assertEqual(ensbuilder.ensemble_nbest,1)
