import shlex
import os
import subprocess


def submit_call(call, log_dir=None):
    print "Calling: " + call
    call = shlex.split(call)
    try:
        if log_dir is None:
            proc = subprocess.Popen(call, stdout=open(os.devnull, 'w'))
        else:
            proc = subprocess.Popen(call, stdout=open(os.path.join(log_dir, "ensemble_out.log"), 'w'),
                                    stderr=open(os.path.join(log_dir, "ensemble_err.log"), 'w'))
        #proc_id = proc.pid
    except OSError as e:
        print e
        return -1
    return proc.pid


def get_algo_exec(runsolver_limit, runsolver_delay, target_call_limit):

    # Create call to autosklearn
    path_to_wrapper = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_wrapper = os.path.join(path_to_wrapper, "wrapper")
    path_to_wrapper = os.path.abspath(path_to_wrapper)
    wrapper_exec = os.path.join(path_to_wrapper, "wrapper_for_SMAC.py")
    if not os.path.exists(wrapper_exec):
        call = 'python wrapper_for_SMAC.py'
    else:
        call = 'python %s' % wrapper_exec
    call += " --limit %d" % target_call_limit

    # Now add runsolver command
    #runsolver_prefix = "runsolver --watcher-data /dev/null -W %d" % \
    #                   (runsolver_limit)
    runsolver_prefix = "runsolver --watcher-data /dev/null -W %d -d %d" % \
                       (runsolver_limit, runsolver_delay)
    call = '"' + runsolver_prefix + " " + call + '"'
    return call


def run_smac(tmp_dir, searchspace, instance_file, limit,
             initial_challengers=None):
    if limit <= 0:
        # It makes no sense to start building ensembles_statistics
        return
    limit = int(limit)
    wallclock_limit = int(limit)
    cutoff_time = int(wallclock_limit/5)
    if cutoff_time < 10:
        # It makes no sense to use less than 10sec
        # We try to do at least one run within the whole runtime
        cutoff_time = int(wallclock_limit) - 5

    cutoff_time_target_function_sees = cutoff_time - 10  # Not needed
    runsolver_softlimit = cutoff_time - 35
    runsolver_hardlimit_delay = 30

    algo_exec = get_algo_exec(runsolver_limit=runsolver_softlimit,
                              runsolver_delay=runsolver_hardlimit_delay,
                              target_call_limit=cutoff_time_target_function_sees)

    if initial_challengers is None:
        initial_challengers = []

    # Bad hack to find smac
    call = os.path.join("smac")
    call = " ".join(['bash', call, '--numRun', '2147483647',
                    '--cli-log-all-calls false',
                    '--cutoffTime', str(cutoff_time),
                    '--wallclock-limit', str(wallclock_limit),
                    '--intraInstanceObj', 'MEAN',
                    '--runObj', 'QUALITY',
                    '--algoExec',  algo_exec,
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
                    '--abort-on-first-run-crash', 'false',
                    '-p', os.path.abspath(searchspace),
                    '--execDir', tmp_dir,
                    '--instances', instance_file] +
                    initial_challengers)
    proc_id = submit_call(call)
    return proc_id

def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit, output_dir):
    if limit <= 0:
        # It makes no sense to start building ensembles_statistics
        return
    path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrapper_exec = os.path.join(path_to_root, "ensemble_script.py")
    runsolver_exec = os.path.join(path_to_root, "lib", "runsolver")
    delay = 5

    call = " ".join(["python", wrapper_exec, tmp_dir, dataset_name,
                        task_type, metric, str(limit-5), output_dir])

    # Now add runsolver command
    #runsolver_cmd = "%s --watcher-data /dev/null -W %d" % \
    #                (runsolver_exec, limit)
    runsolver_cmd = "%s --watcher-data /dev/null -W %d -d %d" % \
                    (runsolver_exec, limit, delay)
    call = runsolver_cmd + " " + call

    proc_id = submit_call(call, log_dir=tmp_dir)
    return proc_id



