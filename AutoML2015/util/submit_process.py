import time
import os

def run_smac(limit):
    time.sleep(2)
    return limit


def run_ensemble_builder(data_dir, dataset_name, limit):
    time.sleep(2)
    os.system("python ensembles.py " + data_dir + dataset_name)
    return limit
