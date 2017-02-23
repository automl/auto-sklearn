# -*- encoding: utf-8 -*-
import builtins
import sys
import unittest
import unittest.mock

import sklearn.tree

from autosklearn.util.backend import Backend


class BackendModelsTest(unittest.TestCase):

    class BackendStub(Backend):

        def __init__(self):
            self.__class__ = Backend

    def setUp(self):
        self.model_directory = '/model_directory/'
        self.backend = self.BackendStub()
        self.backend.get_model_dir = lambda: self.model_directory

    def test_load_models_by_file_names(self):
        self.backend.load_model_by_seed_and_id = unittest.mock.Mock()
        self.backend.load_model_by_seed_and_id.side_effect = lambda *args: args
        rval = self.backend.load_models_by_file_names(['1.2.model',
                                                       '1.3.model',
                                                       '1.4.models'])
        self.assertEqual(rval, {(1, 2): (1, 2),
                                (1, 3): (1, 3)})

    @unittest.mock.patch('pickle.load')
    @unittest.mock.patch('os.path.exists')
    def test_load_model_by_seed_and_id(self, exists_mock, pickleLoadMock):
        exists_mock.return_value = False
        open_mock = unittest.mock.mock_open(read_data='Data')
        with unittest.mock.patch('autosklearn.util.backend.open', open_mock, create=True):
            seed = 13
            idx = 17
            expected_model = self._setup_load_model_mocks(open_mock,
                                                          pickleLoadMock,
                                                          seed, idx)

            actual_model = self.backend.load_model_by_seed_and_id(seed, idx)

            self.assertEqual(expected_model, actual_model)

    @unittest.mock.patch('pickle.load')
    @unittest.mock.patch.object(builtins, 'open')
    @unittest.mock.patch('os.path.exists')
    def test_loads_models_by_identifiers(self, exists_mock, openMock, pickleLoadMock):
        exists_mock.return_value = True
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

        fileMock = unittest.mock.MagicMock()
        fileMock.__enter__.return_value = file_handler

        openMock.side_effect = lambda path, flag: fileMock if path == model_path and flag == 'rb' else None
        pickleLoadMock.side_effect = lambda fh: expected_model if fh == file_handler else None

        return expected_model
