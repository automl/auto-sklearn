# -*- encoding: utf-8 -*-
import os
import glob

import numpy as np


def print_debug_information(automl):

    # In case it is called with estimator,
    # Get the automl object
    if hasattr(automl, 'automl_'):
        automl = automl.automl_

    # Log file path
    log_file = glob.glob(os.path.join(
        automl._backend.temporary_directory, 'AutoML*.log'))[0]

    include_messages = ['INFO', 'DEBUG', 'WARN',
                        'CRITICAL', 'ERROR', 'FATAL']

    # There is a lot of content in the log files. Only
    # parsing the main message and ignore the metalearning
    # messages
    try:
        with open(log_file) as logfile:
            content = logfile.readlines()

        # Get the messages to debug easier!
        content = [x for x in content if any(
            msg in x for msg in include_messages
        ) and 'metalearning' not in x]

    except Exception as e:
        return str(e)

    # Also add the run history if any
    if hasattr(automl, 'runhistory_') and hasattr(automl.runhistory_, 'data'):
        for k, v in automl.runhistory_.data.items():
            content += ["{}->{}".format(k, v)]
    else:
        content += ['No RunHistory']

    # Also add the ensemble history if any
    if len(automl.ensemble_performance_history) > 0:
        content += [str(h) for h in automl.ensemble_performance_history]
    else:
        content += ['No Ensemble History']

    return os.linesep.join(content)


def count_succeses(cv_results):
    return np.sum(
        [status in ['Success', 'Success (but do not advance to higher budget)']
         for status in cv_results['status']]
    )
