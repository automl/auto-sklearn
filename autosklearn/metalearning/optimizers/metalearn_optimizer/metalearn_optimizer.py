##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/env python

import os
import StringIO

import HPOlib.cv as cv
import HPOlib.wrapping_util as wrapping_util

import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as metalearner


def build_metalearn_call(config, options, optimizer_dir):
    cv_file = cv.__file__[:-1] if cv.__file__[-3:] == "pyc" else cv.__file__
    metalearner_file = os.path.abspath(metalearner.__file__)
    if metalearner_file[:-4] == ".pyc":
        metalearner_file = metalearner_file[:-1]

    call = StringIO.StringIO()
    call.write('python ' + metalearner_file)
    call.write(' ' + config.get("EXPERIMENT", "task_args_pkl"))
    call.write(' ' + config.get("METALEARNING", "tasks"))
    call.write(' ' + config.get("METALEARNING", "experiments"))
    call.write(' ' + config.get("EXPERIMENT", "openml_data_dir"))
    call.write(" --distance_measure " + config.get('METALEARNING', 'distance_measure'))
    call.write(" --cli_target " + "'python " + cv_file + "'")
    call.write(" --cwd " + optimizer_dir)
    call.write(" --number_of_jobs " + config.get("HPOLIB", "number_of_jobs"))
    call.write(" --seed " + config.get("HPOLIB", "seed"))
    if config.has_option('METALEARNING', 'distance_keep_features'):
        keep = config.get('METALEARNING', 'distance_keep_features')
        if keep:
            call.write(' --distance_keep_features ' + keep)
    if config.has_option('METALEARNING', 'metafeatures_subset'):
        mf_subset = config.get('METALEARNING', 'metafeatures_subset')
        call.write(' --metafeatures_subset ' + mf_subset)


    # seed = config.getint("HPOLIB", "seed")
    # metric = config.get("METALEARNING", "distance_measure")
    # openml_dir = config.get("EXPERIMENT", "openml_data_dir")
    # rf_params = config.get("METALEARNING", "distance_learner_params")
    # eliminate = config.get("METALEARNING", "distance_eliminate_features")
    # experiments_list_file = config.get("METALEARNING", "experiments")
    # task_file = config.get("EXPERIMENT", "task_args_pkl")
    # tasks_file = config.get("METALEARNING", "tasks")

    return call.getvalue()


def check_dependencies():
    pass


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    raise NotImplementedError()


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()
    cmd = ""

    path_to_optimizer = os.path.abspath(os.path.dirname(metalearner.__file__))

    # Find experiment directory
    if options.restore:
        raise NotImplementedError()
    else:
        optimizer_dir = os.path.join(experiment_dir,
                                     experiment_directory_prefix
                                     + "metalearn_optimizer_" +
                                     str(options.seed) + "_" +
                                     time_string)

    # Build call
    cmd += build_metalearn_call(config, options, optimizer_dir)


    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
    #    params = config.get('METALEARNING', 'params')
    #    if not os.path.exists(os.path.join(optimizer_dir, params)):
    #        os.symlink(os.path.join(experiment_dir, "metalearner", params),
    #            os.path.join(optimizer_dir, params))

    return cmd, optimizer_dir
