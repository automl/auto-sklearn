import shlex
import os
import subprocess


def submit_call(call):
    print "Calling: " + call
    call = shlex.split(call)
    try:
        proc = subprocess.Popen(call, stdout=open(os.devnull, 'w'))
        proc_id = proc.pid
    except OSError as e:
        print e
        return -1
    return proc_id


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
    runsolver_prefix = "runsolver --watcher-data /dev/null -W %d -d %d" % \
                       (runsolver_limit, runsolver_delay)
    call = '"' + runsolver_prefix + " " + call + '"'
    return call


def run_smac(tmp_dir, searchspace, instance_file, limit):
    if limit <= 0:
        # It makes no sense to start building ensembles
        return
    limit = int(limit)
    wallclock_limit = int(limit)
    cutoff_time = int(wallclock_limit/5)
    if cutoff_time < 10:
        # It makes no sense to use less than 10sec
        # We try to do at least one run within the whole runtime
        cutoff_time = int(wallclock_limit) - 5

    cutoff_time_target_function_sees = cutoff_time - 10
    cutoff_time_runsolver_respects = cutoff_time - 5
    cutoff_time_runsolver_delay = 5

    algo_exec = get_algo_exec(runsolver_limit=cutoff_time_runsolver_respects,
                              runsolver_delay=cutoff_time_runsolver_delay,
                              target_call_limit=cutoff_time_target_function_sees)

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
                    '--instances', instance_file])
    pid = submit_call(call)
    return pid


def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit, output_dir):
    if limit <= 0:
        # It makes no sense to start building ensembles
        return
    path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrapper_exec = os.path.join(path_to_root, "ensembles.py")
    runsolver_exec = os.path.join(path_to_root, "lib", "runsolver")
    delay = 5

    call = " ".join(["python", wrapper_exec, tmp_dir, dataset_name,
                        task_type, metric, str(limit-5), output_dir])

    # Now add runsolver command
    runsolver_cmd = "%s --watcher-data /dev/null -W %d -d %d" % \
                    (runsolver_exec, limit, delay)
    call = runsolver_cmd + " " + call

    pid = submit_call(call)
    return pid

