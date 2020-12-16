# -*- encoding: utf-8 -*-
import re
import os
import glob
import typing

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
        content = [line for line in content if any(
            msg in line for msg in include_messages
        ) and 'metalearning' not in line]

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


class AutoMLLogParser(object):
    def __init__(self, logfile: str):
        self.logfile = logfile
        self.lines = self.parse_logfile()

    def parse_logfile(self) -> typing.List[str]:
        # We care about the [debug/info/...] messages
        # At the time of writing, the message format was:
        # [DEBUG] [2020-11-30 11:54:05,072:EnsembleBuilder] Restricting your
        # function to 3072 mb memory.
        #
        # [DEBUG] [2020-11-30 11:53:55,062:pynisher] Redirecting
        # output of the function to files.
        assert os.path.exists(self.logfile), "{} not found".format(self.logfile)

        with open(self.logfile) as fh:
            content = [line.strip() for line in fh if re.search(r'[\w+]', line)]
        return content

    def count_ensembler_iterations(self) -> int:
        iterations = []

        # One thing is to call phynisher, the other is to actually execute the funciton
        iterations_from_inside_ensemble_builder = []
        for line in self.lines:

            # Pynisher call
            # we have to count the start msg from pynisher
            # and the return msg
            # We expect the start msg to be something like:
            # [DEBUG] [2020-11-26 19:22:42,160:EnsembleBuilder] \
            # Function called with argument: (61....
            # [DEBUG] [2020-11-30 11:53:47,069:EnsembleBuilder] Function called with argument:
            # (28.246965646743774, 1, False), {}
            match = re.search(
                r'EnsembleBuilder]\s+Function called with argument:\s+\(\d+\.\d+, (\d+), \w+',
                line)
            if match:
                iterations.append(int(match.group(1)))

            # Ensemble Builder actual call
            # Here we expect the msg:
            # [DEBUG] [2020-11-30 11:53:14,877:EnsembleBuilder] Starting iteration 0,
            # time left: 61.266255
            # [DEBUG] [2020-11-27 20:27:28,044:EnsembleBuilder] Starting iteration 2,
            # time left: 10.603252
            match = re.search(
                r'EnsembleBuilder]\s+Starting iteration (\d+)',
                line)
            if match:
                iterations_from_inside_ensemble_builder.append(int(match.group(1)))

            # The ensemble builder might not be called if there is no time.
            # Here we expect the msg:
            # [DEBUG] [2020-11-27 20:27:28,044:EnsembleBuilder] Not starting iteration 2,
            # as time left: 1.59324
            match = re.search(
                r'EnsembleBuilder]\s+Not starting iteration (\d+)',
                line)
            if match:
                iterations_from_inside_ensemble_builder.append(int(match.group(1)))

        assert iterations == iterations_from_inside_ensemble_builder, "{} ! {}".format(
            iterations, iterations_from_inside_ensemble_builder
        )

        return iterations

    def count_ensembler_success_pynisher_calls(self) -> int:

        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-30 11:53:47,911:EnsembleBuilder] return value:
        # (([{'Timestamp': Timestamp('2020-11-30 11:53:47.910727'),
        # 'ensemble_optimization_score': 0.9787234042553191}], 50, None, None, None), 0)
        # [DEBUG] [2020-11-30 11:54:05,984:EnsembleBuilder] return value:
        # (([{'Timestamp': Timestamp('2020-11- 30 11:54:05.983837'),
        # 'ensemble_optimization_score': 0.9787234042553191}], 50, None, None, None), 0)
        return_msgs = len([line for line in self.lines if re.search(
            r'EnsembleBuilder]\s+return value:.*Timestamp', line)])

        return return_msgs

    def count_tae_pynisher_calls(self) -> int:
        # We expect the return msg to be something like:
        # [DEBUG] [2020-12-16 11:57:08,987:Client-pynisher] Function called with argument: ()
        # , {'queue': <multiprocessing.queues.Queue object at 0x7f9e3cfaae20>, 'config': 1
        # [DEBUG] [2020-12-16 11:57:10,537:Client-pynisher] Function called with argument: ()
        # , {'queue': <multiprocessing.queues.Queue object at 0x7f16f5d95c40>,
        # 'config': Configuration:
        # Only the parenthesis below need to be escaped, ] and { do not.
        call_msgs = len([line for line in self.lines if re.search(
            r'pynisher]\s+Function called with argument: \(\), {', line)])
        return call_msgs

    def count_tae_pynisher_returns(self) -> int:
        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-30 11:53:11,264:pynisher] return value: (None, 0)
        # [DEBUG] [2020-11-30 11:53:13,768:pynisher] return value: (None, 0)
        return_msgs = len([line for line in self.lines if re.search(
            r'pynisher]\s+return value:\s+', line)])
        # When the pynisher pipe is prematurely closed, we also expect:
        # Your function call closed the pipe prematurely
        # -> Subprocess probably got an uncatchable signal
        # We expect the return msg to be something like:
        # OR
        # Something else went wrong, sorry.
        premature_msgs = len([line for line in self.lines if re.search(
            r'pynisher]\s+Your function call closed the pipe prematurely', line)])
        failure_msgs = len([line for line in self.lines if re.search(
            r'pynisher]\s+Something else went wrong, sorry.', line)])
        return return_msgs + premature_msgs + failure_msgs

    def get_automl_setting_from_log(self, dataset_name: str, setting: str) -> str:
        for line in self.lines:
            # We expect messages of the form
            # [DEBUG] [2020-11-30 11:53:10,457:AutoML(5):breast_cancer]   ensemble_size: 50
            # [DEBUG] [2020-11-30 11:53:10,457:AutoML(5):breast_cancer]   ensemble_nbest: 50
            match = re.search(
                f"{dataset_name}]\\s*{setting}\\s*:\\s*(\\w+)",
                line)
            if match:
                return match.group(1)
        return None
