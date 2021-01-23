import os
import shutil
import unittest
import unittest.mock

import numpy as np

from autosklearn.metrics import make_scorer
from autosklearn.ensemble_builder import (
    EnsembleBuilder, AbstractEnsemble
)


def scorer_function(a, b):
    return 0.9


MockMetric = make_scorer('mock', scorer_function)


class BackendMock(object):

    def __init__(self, target_directory):
        this_directory = os.path.abspath(
            os.path.dirname(__file__)
        )
        shutil.copytree(os.path.join(this_directory, 'data'), os.path.join(target_directory))
        self.temporary_directory = target_directory
        self.internals_directory = os.path.join(self.temporary_directory, '.auto-sklearn')

    def load_datamanager(self):
        manager = unittest.mock.Mock()
        manager.__reduce__ = lambda self: (unittest.mock.MagicMock, ())
        array = np.load(os.path.join(
            self.temporary_directory,
            '.auto-sklearn',
            'runs', '0_3_100.0',
            'predictions_test_0_3_100.0.npy'
        ))
        manager.data.get.return_value = array
        return manager

    def load_targets_ensemble(self):
        with open(os.path.join(
            self.temporary_directory,
            ".auto-sklearn",
            "predictions_ensemble_true.npy"
        ), "rb") as fp:
            y = np.load(fp, allow_pickle=True)
        return y

    def save_ensemble(self, ensemble, index_run, seed):
        return

    def save_predictions_as_txt(self, predictions, subset, idx, prefix, precision):
        return

    def get_runs_directory(self) -> str:
        return os.path.join(self.temporary_directory, '.auto-sklearn', 'runs')

    def get_numrun_directory(self, seed: int, num_run: int, budget: float) -> str:
        return os.path.join(self.get_runs_directory(), '%d_%d_%s' % (seed, num_run, budget))

    def get_model_filename(self, seed: int, idx: int, budget: float) -> str:
        return '%s.%s.%s.model' % (seed, idx, budget)


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
        return True

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds: int,
                index_run: int):
        np.ones([10000000, 1000000])
