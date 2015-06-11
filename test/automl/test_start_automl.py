import cPickle
import multiprocessing
import os
import shutil
import unittest

import numpy as np

import autosklearn.start_automl


class AutoMLTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(__file__)
        self.output = os.path.join(self.test_dir, "..", ".tmp")
        try:
            shutil.rmtree(self.output)
        except:
            pass
        os.makedirs(self.output)

    def tearDown(self):
        try:
            shutil.rmtree(self.output)
        except:
            pass

    def test_dataset_manager_pickling(self):
        data_dir = os.path.join(self.test_dir, "..", ".data")
        dataset = "401_bac"
        data_manager_file = os.path.join(self.output, "%s_Manager.pkl" %
                                         dataset)

        queue = multiprocessing.Queue()
        auto = autosklearn.start_automl.AutoML(queue, dataset, data_dir,
                                               self.output, self.output, 60,
                                               60,
                                               initial_configurations_via_metalearning=25)
        auto.start_automl()
        with open(data_manager_file) as fh:
            D = cPickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'].data[:3], [1., 1., 2.]))