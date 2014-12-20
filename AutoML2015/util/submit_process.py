import time
import os


def run_smac(dataset_name, limit):
    time.sleep(2)
    return limit


def run_ensemble_builder(tmp_dir, dataset_name, limit):
    time.sleep(2)
    os.system("python ensembles.py " + tmp_dir + dataset_name)
    return limit
