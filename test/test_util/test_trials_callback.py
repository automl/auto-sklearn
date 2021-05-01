import os
import shutil
import unittest

import pandas as pd

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import StatusType

import autosklearn.pipeline.util as putil
from autosklearn.classification import AutoSklearnClassifier


class AutoMLTrialsCallBack(IncorporateRunResultCallback):

    def __init__(self, output_dir):
        self.trials_num = 1
        self.output_dir = output_dir
        with open(os.path.join(self.output_dir, "trials.csv"), "w") as fp:
            fp.write("TrialNo, StartTime, EndTime, Status, TrainLoss, ValidLoss, TestLoss, Classifier")

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
        with open(os.path.join(self.output_dir, "trials.csv"), "a+") as fp:
            fp.write(f"\n {self.trials_num}, {trial_start_time}, {trial_end_time}, {trial_status}, "
                     f"{train_loss}, {valid_loss}, {test_loss}, {trial_classifier}")
        self.trials_num += 1


class VerifyTrialsCallBack(unittest.TestCase):

    def test_trials_callback_execution(self):
        base_dir = "./tmp"
        tmp_folder = os.path.join(base_dir, "temp")
        output_folder = os.path.join(base_dir, "output")
        os.makedirs("./tmp")
        X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')
        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                   initial_configurations_via_metalearning=0,
                                   per_run_time_limit=10,
                                   memory_limit=1024,
                                   tmp_folder=tmp_folder,
                                   delete_tmp_folder_after_terminate=False,
                                   output_folder=output_folder,
                                   delete_output_folder_after_terminate=False,
                                   n_jobs=1,
                                   include_estimators=["sgd"],
                                   include_preprocessors=["no_preprocessing"],
                                   get_trials_callback=AutoMLTrialsCallBack(base_dir)
                                   )
        cls.fit(X_train, Y_train, X_test,Y_test)
        trials = pd.read_csv(os.path.join(base_dir, "trials.csv"))
        assert trials.shape[0] > 1
        assert len(os.listdir(output_folder)) > 0
        shutil.rmtree(base_dir)

    def test_no_trials_callback(self):
        base_dir = "./tmp"
        tmp_folder = os.path.join(base_dir, "temp")
        output_folder = os.path.join(base_dir, "output")
        os.makedirs("./tmp")
        X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')
        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                   initial_configurations_via_metalearning=0,
                                   per_run_time_limit=10,
                                   memory_limit=1024,
                                   tmp_folder=tmp_folder,
                                   delete_tmp_folder_after_terminate=False,
                                   output_folder=output_folder,
                                   delete_output_folder_after_terminate=False,
                                   n_jobs=1,
                                   include_estimators=["sgd"],
                                   include_preprocessors=["no_preprocessing"]
                                   )
        cls.fit(X_train, Y_train, X_test,Y_test)
        assert len(os.listdir(output_folder)) > 0
        shutil.rmtree(base_dir)