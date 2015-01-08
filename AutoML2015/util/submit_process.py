import shlex
import os
import subprocess


def submit_call(call):
    print "Calling: " + call
    call = shlex.split(call)
    try:
        proc = subprocess.Popen(call, stdout=open(os.devnull, 'w'))
        proc_id = proc.pid()
    except OSError as e:
        print e
        return -1
    return proc_id


def get_algo_exec():
    # Create call to autosklearn
    path_to_wrapper = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrapper_exec = os.path.join(path_to_wrapper, "run_config_evaluation.py")
    if not os.path.exists(wrapper_exec):
        call = '"python run_config_evaluation.py"'
    else:
        call = '"python %s"' % wrapper_exec
    return call


def run_smac(tmp_dir, searchspace, instance_file, limit):
    if limit <= 0:
        # It makes no sense to start building ensembles
        return
    # Bad hack to find smac
    call = os.path.join("smac")
    call = " ".join([call, '--numRun', '2147483647',
                    '--cli-log-all-calls false',
                    '--cutoffTime', '2147483647',
                    '--wallclock-limit', '10',
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
    pid = submit_call(call)
    return pid


def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit, output_dir):
    if limit <= 0:
        # It makes no sense to start building ensembles
        return
    path_to_wrapper = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrapper_exec = os.path.join(path_to_wrapper, "ensembles.py")
    call = " ".join(["python", wrapper_exec, tmp_dir, dataset_name,
                        task_type, metric, str(limit), output_dir])
    pid = submit_call(call)
    return pid

