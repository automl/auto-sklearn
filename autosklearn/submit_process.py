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
    return proc


def get_algo_exec(runsolver_limit, runsolver_delay):

    # Create call to autosklearn
    path_to_wrapper = os.path.dirname(os.path.abspath(__file__))
    wrapper_exec = os.path.join(path_to_wrapper, "wrapper_for_SMAC.py")
    call = 'python %s' % wrapper_exec

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    runsolver_limit = max(1, runsolver_limit)

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

    runsolver_softlimit = cutoff_time - 35
    runsolver_hardlimit_delay = 30

    algo_exec = get_algo_exec(runsolver_limit=runsolver_softlimit,
                              runsolver_delay=runsolver_hardlimit_delay,)

    if initial_challengers is None:
        initial_challengers = []

    # Bad hack to find smac
    call = os.path.join("smac")
    call = " ".join([call, '--numRun', '2147483647',
                    '--cli-log-all-calls false',
                    '--console-log-level DEBUG',
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
    proc = submit_call(call)
    return proc


def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit,
                         output_dir, ensemble_size):
    if limit <= 0:
        # It makes no sense to start building ensembles_statistics
        return
    path_to_root = os.path.dirname(os.path.abspath(__file__))
    wrapper_exec = os.path.join(path_to_root, "ensemble_selection_script.py")
    runsolver_exec = "runsolver"
    delay = 5

    call = " ".join(["python", wrapper_exec, tmp_dir, dataset_name,
                        task_type, metric, str(limit-5), output_dir,
                        str(ensemble_size)])

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    limit = max(1, limit)

    # Now add runsolver command
    #runsolver_cmd = "%s --watcher-data /dev/null -W %d" % \
    #                (runsolver_exec, limit)
    runsolver_cmd = "%s --watcher-data /dev/null -W %d -d %d" % \
                    (runsolver_exec, limit, delay)
    call = runsolver_cmd + " " + call

    proc = submit_call(call, log_dir=tmp_dir)
    return proc



