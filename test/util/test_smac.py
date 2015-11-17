from __future__ import print_function
import unittest

import mock

import autosklearn.util.smac
from autosklearn.util import Backend


class SMACTest(unittest.TestCase):
    @mock.patch.object(Backend, 'write_txt_file')
    def test_write_instance_file(self, MockedBackend):
        # Holdout
        backend = MockedBackend(None, None)
        autosklearn.util.smac._write_instance_file(
            'holdout', None, 'path', backend, 'tmp_dir'
        )
        self.assertEqual(len(backend.method_calls), 2)
        self.assertEqual(backend.method_calls[0][1],
                         ('tmp_dir/instances.txt', 'holdout path', 'Instances'))
        self.assertEqual(backend.method_calls[1][1],
                         ('tmp_dir/test_instances.txt', 'test path',
                          'Test instances'))

        # Nested CV
        backend = MockedBackend(None, None)
        autosklearn.util.smac._write_instance_file(
            'nested-cv', {'inner_folds': 2, 'outer_folds': 2},
            'path', backend, 'tmp_dir'
        )
        self.assertEqual(len(backend.method_calls), 4)
        self.assertEqual(backend.method_calls[2][1],
                         ('tmp_dir/instances.txt', 'nested-cv:2/2 path',
                          'Instances'))
        self.assertEqual(backend.method_calls[3][1],
                         ('tmp_dir/test_instances.txt', 'test path',
                          'Test instances'))

        # CV
        backend = MockedBackend(None, None)
        autosklearn.util.smac._write_instance_file(
            'cv', {'folds': 2},
            'path', backend, 'tmp_dir'
        )
        self.assertEqual(len(backend.method_calls), 6)
        self.assertEqual(backend.method_calls[4][1],
                         ('tmp_dir/instances.txt', 'cv:2 path',
                          'Instances'))
        self.assertEqual(backend.method_calls[5][1],
                         ('tmp_dir/test_instances.txt', 'test path',
                          'Test instances'))

        # Partial-CV
        backend = MockedBackend(None, None)
        autosklearn.util.smac._write_instance_file(
            'partial-cv', {'folds': 2},
            'path', backend, 'tmp_dir'
        )
        self.assertEqual(len(backend.method_calls), 8)
        self.assertEqual(backend.method_calls[6][1],
                         ('tmp_dir/instances.txt',
                          'partial-cv:0/2 path\npartial-cv:1/2 path',
                          'Instances'))
        self.assertEqual(backend.method_calls[7][1],
                         ('tmp_dir/test_instances.txt', 'test path',
                          'Test instances'))