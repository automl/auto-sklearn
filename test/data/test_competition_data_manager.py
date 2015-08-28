import os
import unittest

from autosklearn.data.competition_data_manager import CompetitionDataManager


class CompetitionDataManagerTest(unittest.TestCase):
    def setUp(self):
        self.path_to_dataset = os.path.join(
            os.path.dirname(__file__), '..', '.data', '31_bac')
