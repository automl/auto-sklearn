# -*- encoding: utf-8 -*-
from __future__ import print_function
import unittest
import unittest.mock

from autosklearn.automl import AutoML
from autosklearn.util.backend import Backend


class AutoMLStub(AutoML):

    def __init__(self):
        self.__class__ = AutoML


class AutoMlModelsTest(unittest.TestCase):

    def setUp(self):
        self.automl = AutoMLStub()
        self.automl._shared_mode = False
        self.automl._seed = 42
        self.automl._backend = unittest.mock.Mock(spec=Backend)
        self.automl._delete_output_directories = lambda: 0

    def test_only_loads_ensemble_models(self):
        identifiers = [(1, 2), (3, 4)]
        models = [ 42 ]
        self.automl._backend.load_ensemble.return_value.identifiers_ \
            = identifiers
        self.automl._backend.load_models_by_identifiers.side_effect \
            = lambda ids: models if ids is identifiers else None

        self.automl._load_models()

        self.assertEqual(models, self.automl.models_)

    def test_loads_all_models_if_no_ensemble(self):
        models = [ 42 ]
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.load_all_models.return_value = models

        self.automl._load_models()

        self.assertEqual(models, self.automl.models_)

    def test_raises_if_no_models(self):
        self.automl._backend.load_ensemble.return_value = None
        self.automl._backend.load_all_models.return_value = []

        self.assertRaises(ValueError, self.automl._load_models)
