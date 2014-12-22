import time
import os
import sys


def get_algo_exec():
    # Create call to autosklearn
    call = "CALL_AUTOSKLEARN"
    return call


def run_smac(tmp_dir, dataset_name, searchspace, instance_file, limit):
    call = "/home/eggenspk/HPOlib/HPOlib/optimizers/smac/smac_2_08_00-master_src/smac"
    call = " ".join([call, '--numRun', '2147483647',
                    '--cli-log-all-calls false',
                    '--cutoffTime', '2147483647',
                    '--intraInstanceObj', 'MEAN',
                    '--runObj', 'QUALITY',
                    '--algoExec',  get_algo_exec(),
                    '--numIterations', '2147483647',
                    '--totalNumRunsLimit', '2147483647',
                    '--outputDirectory', tmp_dir,
                    '--numConcurrentAlgoExecs', '1',
                    '--maxIncumbentRuns', '2147483647',
                    '--retryTargetAlgorithmRunCount', '0',
                    '--intensification-percentage', '0',
                    '--initial-incumbent', 'DEFAULT',
                    '--rf-split-min', '10',
                    '--validation', 'false',
                    '--deterministic', 'true',
                    '-p', os.path.abspath(searchspace),
                    '--execDir', tmp_dir,
                    '--instances', instance_file])
    print call
    time.sleep(2)
    return limit


def run_ensemble_builder(tmp_dir, dataset_name, limit):
    time.sleep(2)
    os.system("python ensembles.py " + tmp_dir + dataset_name)
    return limit
