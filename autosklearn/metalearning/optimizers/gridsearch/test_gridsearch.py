from collections import OrderedDict
import unittest
import os

import gridsearch.gridsearch as gridsearch


class GridSearchTest(unittest.TestCase):
    def setUp(self):
        self.hyperparameters = OrderedDict()
        self.hyperparameters["x"] = [-5, 0, 5, 10]
        self.hyperparameters["y"] = [0, 5, 10, 15]

    def test_build_grid(self):
        grid = gridsearch.build_grid(self.hyperparameters)
        self.assertIsInstance(grid, list)
        self.assertEqual(len(grid), 16)
        self.assertDictEqual(grid[0], {"x": -5, "y": 0})
        self.assertDictEqual(grid[3], {"x": -5, "y": 15})
        self.assertDictEqual(grid[9], {"x": 5, "y": 5})
        self.assertDictEqual(grid[15], {"x": 10, "y": 15})

    def test_build_grid_3d(self):
        self.hyperparameters["z"] = [5, 10, 15, 20]
        grid = gridsearch.build_grid(self.hyperparameters)
        self.assertIsInstance(grid, list)
        self.assertEqual(len(grid), 64)
        self.assertDictEqual(grid[0], {"x": -5, "y": 0, "z": 5})
        self.assertDictEqual(grid[16], {"x": 0, "y": 0, "z": 5})

    def test_perform_grid_search(self):
        from numpy import pi, cos

        def branin(params):
            x = params["x"]
            y = params["y"]
            result = (y-(5.1/(4*pi**2))*x**2+5*x/pi-6)**2
            result += 10*(1-1/(8*pi))*cos(x)+10
            return result

        grid = gridsearch.build_grid(self.hyperparameters)
        gridsearch.perform_gridsearch(branin, grid)

