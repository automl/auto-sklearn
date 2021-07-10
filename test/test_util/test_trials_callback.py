import os
import tempfile
import unittest

import pandas as pd
from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import StatusType

import autosklearn.pipeline.util as putil
from autosklearn.classification import AutoSklearnClassifier


class AutoMLTrialsCallBack(IncorporateRunResultCallback):

    def __init__(self, fname):
        self.trials_num = 1
        self.fname = fname
        with open(fname, "w") as fp:
            fp.write("TrialNo, "
                     "StartTime, "
                     "EndTime, "
                     "Status, "
                     "TrainLoss, "
                     "ValidLoss, "
                     "TestLoss, "
                     "Classifier")

    def __call__(
            self, smbo: 'SMBO',
            run_info: RunInfo,
            result: RunValue,
            time_left: float,
    ) -> None:
        train_loss, valid_loss, test_loss = None, None, None
        trial_start_time = result.starttime
        trial_end_time = result.endtime
        trial_status = result.status.name
        if trial_status == StatusType.SUCCESS.name:
            train_loss = result.additional_info.get('train_loss')
            valid_loss = result.cost
            test_loss = result.additional_info.get('test_loss')
        trial_classifier = run_info.config.get_dictionary()['classifier:__choice__']
        with open(self.fname, "a+") as fp:
            fp.write(f"\n {self.trials_num}, {trial_start_time}, {trial_end_time}, {trial_status}, "
                     f"{train_loss}, {valid_loss}, {test_loss}, {trial_classifier}")
        self.trials_num += 1


class VerifyTrialsCallBack(unittest.TestCase):

    def test_trials_callback_execution(self):
        trials_summary_fname = os.path.join(tempfile.gettempdir(), "trials.csv")
        X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')
        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                    initial_configurations_via_metalearning=0,
                                    per_run_time_limit=10,
                                    memory_limit=1024,
                                    delete_tmp_folder_after_terminate=False,
                                    n_jobs=1,
                                    include={'feature_preprocessor': ['pca'],
                                             'classifier': ['sgd']},
                                    get_trials_callback=AutoMLTrialsCallBack(trials_summary_fname)
                                    )
        cls.fit(X_train, Y_train, X_test, Y_test)
        trials = pd.read_csv(trials_summary_fname)
        assert trials.shape[0] > 0, f"Auto-Sklearn explored {trials.shape[0] - 1} trials"
