import os
import unittest


import numpy as np

from autosklearn.metrics import make_scorer
from autosklearn.ensemble_builder import (
    EnsembleBuilder,
)

this_directory = os.path.dirname(__file__)


def scorer_function(a, b):
    return 0.9


MockMetric = make_scorer('mock', scorer_function)


class BackendMock(object):

    def __init__(self):
        this_directory = os.path.abspath(
            os.path.dirname(__file__)
        )
        self.temporary_directory = os.path.join(
            this_directory, 'data',
        )
        self.internals_directory = os.path.join(
            this_directory, 'data', '.auto-sklearn',
        )

    def load_datamanager(self):
        manager = unittest.mock.Mock()
        manager.__reduce__ = lambda self: (unittest.mock.MagicMock, ())
        array = np.load(os.path.join(
            this_directory, 'data',
            '.auto-sklearn',
            'predictions_test',
            'predictions_test_0_3_100.0.npy'
        ))
        manager.data.get.return_value = array
        return manager

    def load_targets_ensemble(self):
        with open(os.path.join(
            self.temporary_directory,
            ".auto-sklearn",
            "predictions_ensemble",
            "predictions_ensemble_true.npy"
        ), "rb") as fp:
            y = np.load(fp, allow_pickle=True)
        return y

    def get_done_directory(self):
        return os.path.join(this_directory, 'data', '.auto-sklearn', 'done')

    def save_ensemble(self, ensemble, index_run, seed):
        return

    def save_predictions_as_txt(self, predictions, subset, idx, prefix, precision):
        return


def compare_read_preds(read_preds1, read_preds2):
    """
    compares read_preds attribute. An alternative to
    assert Dict Equal as it contains np arrays, so we have
    to use np testing utilities accordingly
    """

    # Both arrays should have the same splits
    assert set(read_preds1.keys()) == set(read_preds2.keys())

    for k, v in read_preds1.items():

        # Each split should have the same elements
        assert set(read_preds1[k].keys()) == set(read_preds2[k].keys())

        # This level contains the scores/ensmebles/etc
        for actual_k, actual_v in read_preds1[k].items():

            # If it is a numpy array, make sure it is the same
            if type(actual_v) is np.ndarray:
                np.testing.assert_array_equal(actual_v, read_preds2[k][actual_k])
            else:
                assert actual_v == read_preds2[k][actual_k]


class EnsembleBuilderMemMock(EnsembleBuilder):

    def fit_ensemble(self, selected_keys):
        np.ones([10000000, 1000000])
