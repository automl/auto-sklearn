# -*- encoding: utf-8 -*-
from __future__ import print_function
import unittest
import mock
from autosklearn.util.backend import Backend

from sys import version_info
if version_info.major == 2:
    import __builtin__ as builtins
else:
    import builtins


class BackendModelsTest(unittest.TestCase):

    class BackendStub(Backend):

        def __init__(self):
            self.__class__ = Backend

    def setUp(self):
        self.model_directory = '/model_directory/'
        self.backend = self.BackendStub()
        self.backend.get_model_dir = lambda: self.model_directory

    @mock.patch('six.moves.cPickle.load')
    @mock.patch.object(builtins, 'open')
    def test_loads_model_by_seed_and_id(self, openMock, pickleLoadMock):
        seed = 13
        idx = 17
        expected_model = self._setup_load_model_mocks(openMock, pickleLoadMock, seed, idx)

        actual_model = self.backend.load_model_by_seed_and_id(seed, idx)

        self.assertEqual(expected_model, actual_model)

    @mock.patch('six.moves.cPickle.load')
    @mock.patch.object(builtins, 'open')
    def test_loads_models_by_identifiers(self, openMock, pickleLoadMock):
        seed = 13
        idx = 17
        expected_model = self._setup_load_model_mocks(openMock, pickleLoadMock, seed, idx)
        expected_dict = { (seed, idx): expected_model }

        actual_dict = self.backend.load_models_by_identifiers([(seed, idx)])

        self.assertIsInstance(actual_dict, dict)
        self.assertDictEqual(expected_dict, actual_dict)

    def _setup_load_model_mocks(self, openMock, pickleLoadMock, seed, idx):
        model_path = '/model_directory/%s.%s.model' % (seed, idx)
        file_handler = 'file_handler'
        expected_model = 'model'

        fileMock = mock.MagicMock()
        fileMock.__enter__.return_value = file_handler

        openMock.side_effect = lambda path, flag: fileMock if path == model_path and flag == 'rb' else None
        pickleLoadMock.side_effect = lambda fh: expected_model if fh == file_handler else None

        return expected_model
