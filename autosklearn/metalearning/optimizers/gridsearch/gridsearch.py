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

import HPOlib.cv as cv
import HPOlib.wrapping_util as wrapping_util
import HPOlib.optimizers.gridsearch.gridsearch.gridsearch as gridsearch


version_info = ("Just a local test version")
__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

optimizer_str = "gridsearch"


def build_gridsearch_call(config, options, optimizer_dir):
    cv_file = cv.__file__[:-1] if cv.__file__[-3:] == "pyc" else cv.__file__
    gridsearch_file = os.path.abspath(gridsearch.__file__)
    if gridsearch_file[:-3] == "pyc":
        gridsearch_file = gridsearch_file[:-1]

    call = "cd " + optimizer_dir + "\n" + "python " + gridsearch_file
    call = ' '.join([call, '--params', config.get('GRIDSEARCH', 'params'),
                     "--cli_target", "'python", cv_file + "'"])
    return call


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    raise NotImplementedError()


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()
    cmd = ""

    path_to_optimizer = os.path.abspath(os.path.dirname(gridsearch.__file__))

    # Find experiment directory
    if options.restore:
        raise NotImplementedError()
    else:
        optimizer_dir = os.path.join(experiment_dir, optimizer_str + "_" +
                                     str(options.seed) + "_" +
                                     time_string)

    # Build call
    cmd += build_gridsearch_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        params = config.get('GRIDSEARCH', 'params')
        # Copy the SMAC search space
        if not os.path.exists(os.path.join(optimizer_dir, params)):
            os.symlink(os.path.join(experiment_dir, optimizer_str, params),
            os.path.join(optimizer_dir, params))

    return cmd, optimizer_dir
