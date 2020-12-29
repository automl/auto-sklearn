import logging
import unittest
import numpy as np

import pandas as pd
from autosklearn.metalearning.metalearning.kNearestDatasets.kND import KNearestDatasets
from autosklearn.metalearning.metalearning.metrics.misc import get_random_metric


class kNDTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                                 "number_of_features": 38.}, name=232)
        self.krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                                 2., "number_of_features": 36.}, name=233)
        self.labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                                2., "number_of_features": 16.}, name=234)
        self.runs = {232: [0.1, 0.5, 0.7],
                     233: [np.NaN, 0.1, 0.7],
                     234: [0.5, 0.7, 0.1]}
        self.runs = pd.DataFrame(self.runs)
        self.logger = logging.getLogger()

    def test_fit_l1_distance(self):
        kND = KNearestDatasets(logger=self.logger)

        kND.fit(pd.DataFrame([self.anneal, self.krvskp, self.labor]), self.runs)
        self.assertEqual(kND.best_configuration_per_dataset[232], 0)
        self.assertEqual(kND.best_configuration_per_dataset[233], 1)
        self.assertEqual(kND.best_configuration_per_dataset[234], 2)
        self.assertTrue((kND.metafeatures ==
                         pd.DataFrame([self.anneal, self.krvskp, self.labor])).all().all())

    # TODO: rename to kNearestTasks or something
    def test_kNearestDatasets(self):
        kND = KNearestDatasets(logger=self.logger)
        kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                self.runs.loc[:, [233, 234]])

        neighbor = kND.kNearestDatasets(self.anneal, 1)
        self.assertEqual([233], neighbor)
        neighbor, distance = kND.kNearestDatasets(self.anneal, 1,
                                                  return_distance=True)
        self.assertEqual([233], neighbor)
        np.testing.assert_array_almost_equal([3.8320802803440586], distance)

        neighbors = kND.kNearestDatasets(self.anneal, 2)
        self.assertEqual([233, 234], neighbors)
        neighbors, distance = kND.kNearestDatasets(self.anneal, 2,
                                                   return_distance=True)
        self.assertEqual([233, 234], neighbors)
        np.testing.assert_array_almost_equal([3.8320802803440586, 4.367919719655942], distance)

        neighbors = kND.kNearestDatasets(self.anneal, -1)
        self.assertEqual([233, 234], neighbors)
        neighbors, distance = kND.kNearestDatasets(self.anneal, -1,
                                                   return_distance=True)
        self.assertEqual([233, 234], neighbors)
        np.testing.assert_array_almost_equal([3.8320802803440586, 4.367919719655942], distance)

        self.assertRaises(ValueError, kND.kNearestDatasets, self.anneal, 0)
        self.assertRaises(ValueError, kND.kNearestDatasets, self.anneal, -2)

    def test_kBestSuggestions(self):
        kND = KNearestDatasets(logger=self.logger)
        kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                self.runs.loc[:, [233, 234]])
        neighbor = kND.kBestSuggestions(self.anneal, 1)
        np.testing.assert_array_almost_equal(
            [(233, 3.8320802803440586, 1)],
            neighbor,
        )
        neighbors = kND.kBestSuggestions(self.anneal, 2)
        np.testing.assert_array_almost_equal(
            [(233, 3.8320802803440586, 1), (234, 4.367919719655942, 2)],
            neighbors,
        )
        neighbors = kND.kBestSuggestions(self.anneal, -1)
        np.testing.assert_array_almost_equal(
            [(233, 3.8320802803440586, 1), (234, 4.367919719655942, 2)],
            neighbors,
        )

        self.assertRaises(ValueError, kND.kBestSuggestions, self.anneal, 0)
        self.assertRaises(ValueError, kND.kBestSuggestions, self.anneal, -2)

    def test_random_metric(self):
        kND = KNearestDatasets(logger=self.logger,
                               metric=get_random_metric(random_state=1))
        kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                self.runs.loc[:, [233, 234]])
        distances = []
        for i in range(20):
            neighbor = kND.kBestSuggestions(self.anneal, 1)
            distances.append(neighbor[0][1])
        self.assertEqual(len(np.unique(distances)), 20)
