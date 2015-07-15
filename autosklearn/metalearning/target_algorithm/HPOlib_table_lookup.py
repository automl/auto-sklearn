from collections import OrderedDict
import cPickle
import time

import numpy as np

import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util

def main(params, **kwargs):
    """Table lookup target algorithm for HPOlib result files.

    This script can be used as a hyperparameter optimization benchmark inside
    the HPOlib after a lookup table was populated, e.g. by a gridsearch. This
    script behaves like the original benchmark, except that it throws an error
    message if an unknown hyperparameter configuration is passed for evaluation.

    The calling conventions are the same as for every other HPOlib benchmark.
    The HPOlib log file can either be specified in the HPOlib configuration file
    in a section EXPERIMENT with the key lookup_table or with the command line
    argument --lookup_table."""

    # Get the lookup table
    if 'lookup_table' in kwargs:
        lookup_table = kwargs['lookup_table']
        pickle_file = open(lookup_table)
    else:
        config = wrapping_util.load_experiment_config_file()
        lookup_table = config.get("EXPERIMENT", "lookup_table")
        pickle_file = open(lookup_table)

    results_pickle = cPickle.load(pickle_file)
    pickle_file.close()

    print "Read experiment pickle %s" % lookup_table
    print "Going to evaluate hyperparameter configuration %s" % params

    folds = int(kwargs["folds"])
    fold = int(kwargs["fold"])
    ground_truth = dict()
    measured_times = dict()

    for trial in results_pickle["trials"]:
        if not np.isfinite(trial["result"]) and \
            np.isfinite(trial["instance_results"]).all():
                raise ValueError("Results and instance_results must be valid "
                                 "numbers.")

        # convert everything to a string since the params which we receive
        # are strings, too
        trial_params = trial["params"]
        for key in trial_params:
            trial_params[key] = str(trial_params[key])

        parameters = str(OrderedDict(sorted(trial["params"].items(),
                                        key=lambda t: t[0])))
        if folds > 1:
            ground_truth[parameters] = trial["instance_results"][fold]
            measured_times[parameters] = trial["instance_durations"][fold]
        else:
            ground_truth[parameters] = trial["result"]
            measured_times[parameters] = trial["duration"]

            # The HPOlib chokes on duration beeing NaN, also this might not
            # be true if we actually ran some of the instances...
            if not np.isfinite(measured_times[parameters]):
                # First, try the sum of instance durations, if this is still
                # NaN return the highest value in the experiment pickle
                instance_durations = np.nansum(trial['instance_durations'])
                if np.isfinite(instance_durations):
                    measured_times[parameters] = instance_durations
                else:
                    longest_run = np.nanmax([trial_['duration'] for trial_ in
                                             results_pickle['trials']])
                    measured_times[parameters] = longest_run


    # Work around HPOlib bug https://github.com/automl/HPOlib/issues/62
    params_hack = dict()
    for key in params:
        params_hack["-" + key] = params[key]

    params = str(OrderedDict(sorted(params_hack.items(), key=lambda t: t[0])))
    y = ground_truth[params]
    runtime = measured_times[params]

    print "Found result", y, "measured runtime was", runtime
    if 'return_runtime' in kwargs:
        return y, runtime
    return y

if __name__ == "__main__":
    args, params = benchmark_util.parse_cli()
    result, runtime = main(params, return_runtime=True, **args)
    # TODO return the time which was measured in the original pickle file!
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", runtime, result, -1, str(__file__))
