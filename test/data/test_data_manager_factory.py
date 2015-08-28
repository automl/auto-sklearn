import os
import unittest

import autosklearn.data.data_manager_factory as factory


class NameSpace(object):
    def __init__(self, dataset, data_format, task=None, metric=None,
                 target=None):
        self.dataset = dataset
        self.data_format = data_format
        self.task = task
        self.metric = metric
        self.target = target


class DataManagerFactoryTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', '.data')

    def test_competition_format(self):
        dataset = os.path.join(self.data_dir, "31_bac")
        namespace = NameSpace(dataset, "automl-competition-format")
        D = factory.get_data_manager(namespace)
        print D

    def test_arff_format(self):
        dataset = os.path.join(self.data_dir, "germancredit")
        namespace = NameSpace(dataset, 'arff',
                              task='binary.classification',
                              metric='acc_metric',
                              target='class')
        D = factory.get_data_manager(namespace)
        print D