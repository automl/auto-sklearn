#!/usr/bin/env python

#############################
# ChaLearn AutoML challenge #
#############################

# Usage: python run.py input_dir output_dir

# This sample code can be used either 
# - to submit RESULTS depostited in the res/ subdirectory or 
# - as a template for CODE submission.
#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_test_000.predict            -- Provide predictions at regular intervals to make sure you get some results even if the program crashes
# 	dataname_test_001.predict
# 	dataname_test_002.predict
# 	...
# 	dataname_valid_000.predict
# 	dataname_valid_001.predict
# 	dataname_valid_002.predict
# 	...
# 
# Result submission:
# =================
# Search for @RESULT to locate that part of the code.
# ** Always keep this code. **
# If the subdirectory res/ contains result files (predicted values)
# the code just copies them to the output and does not train/test models.
# If no results are found, a model is trained and tested (see code submission).
#
# Code submission:
# ===============
# Search for @CODE to locate that part of the code.
# ** You may keep or modify this template or subtitute your own code. **
# The program saves predictions regularly. This way the program produces
# at least some results if it dies (or is terminated) prematurely. 
# This also allows us to plot learning curves. The last result is used by the
# scoring program.
# We implemented 2 classes:
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013

# =========================== BEGIN USER OPTIONS ==============================

# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Sample model:
####################
# True: run the sample model from the starter kit
# False: run the own code
# sample_model = False # -> We always use our model

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 30

# Maximum number of cycles
##########################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
max_cycle = 20

# ZIP your code
###############
# You can create a code submission archive, ready to submit, with zipme = True.
# This is meant to be used on your LOCAL server.
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
submission_filename = 'automl_sample_submission_' + the_date

# I/O defaults
##############
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "/Users/isabelleguyon/Documents/Projects/Codalab/AutoMLcompetition/StartingKit/"
default_input_dir = root_dir + "sample_input0" # "scoring_input0/dorothea"
default_output_dir = root_dir + "scoring_input0/res"

# =========================== END USER OPTIONS ================================

# Version of the sample code
version = 1 

# General purpose functions
import os
from sys import argv, path
import numpy as np
import time
loading_overhead = time.time()
overall_start = time.clock()

# Our directories
# Note: On cadalab, there is an extra sub-directory called "program"

running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir=codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append(run_dir)
path.append(lib_dir)

# ===================================
# ========== THIS IS WHERE WE START =
# ===================================

# === Add libraries to path
import sys
import errno
import multiprocessing
import Queue
import psutil
import pwd
import re

our_root_dir = os.path.abspath(os.path.dirname(__file__))
our_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
smac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "smac-v2.08.00-master-731"))
java_path = os.path.join(our_lib_dir, "jre1.8.0_25", "bin")

# To use it within this scope:
sys.path.insert(0, our_lib_dir)
sys.path.insert(0, our_root_dir)

# Insert our library path to PYTHONPATH
if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] = our_lib_dir + os.pathsep + our_root_dir + os.environ["PYTHONPATH"]

if "PATH" not in os.environ:
    os.environ["PATH"] = ""
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + smac_path +\
                     os.pathsep + our_lib_dir + os.pathsep + java_path
os.environ["JAVA_HOME"] = os.path.join(java_path, "java")

# Imports from this library
import data.data_io as data_io            # general purpose input/output functions
from data.data_io import vprint           # print only in verbose mode
from util import Stopwatch, get_dataset_info, check_pid, check_system_info, util
import start_automl

if debug_mode >= 4 or running_on_codalab: # Show library version and directory structure
    data_io.show_version()
    data_io.show_dir(run_dir)

# =========================== BEGIN PROGRAM ================================
if __name__=="__main__" and debug_mode<4:
    # ==========================================================================
    # ============ CHECK THIS SECTION BEFORE SUBMITTING ========================
    # == Definitions
    BUFFER = 35  # time-left - BUFFER = timelimit for SMAC/ensembles.py
    BUFFER_BEFORE_SENDING_SIGTERM = 30  # We send SIGTERM to all processes
    DELAY_TO_SIGKILL = 15  # And after a delay we send a sigkill

    # == Change permissions for our libraries
    util.change_permission_folder(perm="755", fl=lib_dir)

    # == Check system
    check_system_info.check_system_info()

    # == Some variables
    pid_dict = dict()
    info_dict = dict()

    # == Calc time already spent
    stop_load = time.time()
    loading_overhead = stop_load - loading_overhead
    overall_limit = 0-loading_overhead

    vprint(verbose, "Loading took %f sec" % loading_overhead)
    stop = Stopwatch.StopWatch()
    stop.start_task("wholething")
    stop.start_task("inventory")

    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2])

    # == Create TMP directory to store our output
    tmp_dir = os.path.abspath(os.path.join(output_dir, "TMP"))
    vprint(verbose, "Creating temporary output dir %s" % tmp_dir)

    if not os.path.isdir(tmp_dir):
        try:
            os.mkdir(tmp_dir)
        except os.error as e:
            print(e)
            vprint(verbose, "Using /tmp/ as output directory")
            tmp_dir = "/tmp/"
    TMP_DIR = tmp_dir

    output_dir_list = str(os.listdir(output_dir))
    vprint(verbose, "Output directory contains: %s" % output_dir_list)

    #### Move old results and create a new output directory
    data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_io(input_dir, output_dir)
        print('\n****** Sample code version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = []  # Do not proceed with learning and testing

    stop.stop_task("inventory")
    stop.start_task("submitresults")
    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames) > 0:
        vprint(verbose,  "************************************************************************")
        vprint(verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint(verbose,  "************************************************************************")
        OK = data_io.copy_results(datanames, res_dir, output_dir, verbose)  # DO NOT REMOVE!
        if OK: 
            vprint(verbose,  "[+] Success")
            datanames = []  # Do not proceed with learning and testing
        else:
            vprint(verbose, "======== Some missing results on current datasets!")
            vprint(verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================
    stop.stop_task("submitresults")

    stop.start_task("submitcode")
    # ================ @CODE SUBMISSION (SUBTITUTE YOUR CODE) =================

    # == Get info from all datasets to get overall time limit
    stop.start_task("get_info")
    for basename in datanames:
        info_dict[basename] = get_dataset_info.getInfoFromFile(input_dir, basename)
        overall_limit += info_dict[basename]["time_budget"]

    spent = stop.wall_elapsed("wholething")
    vprint(verbose, "Overall time limit is %f, we already spent %f, left %f" %
           (overall_limit, spent, overall_limit-spent))
    stop.stop_task("get_info")

    # == Loop over all datasets and start AutoML
    queue_dict = dict()
    for basename in datanames:
        stop.start_task(basename)

        vprint(verbose, "************************************************")
        vprint(verbose, "******** Processing dataset " + basename.capitalize() + " ********")
        vprint(verbose, "************************************************")

        # = Make tmp output directory for this dataset
        tmp_dataset_dir = os.path.join(TMP_DIR, basename)
        vprint(verbose, "Makedir %s" % tmp_dataset_dir)
        try:
            os.makedirs(tmp_dataset_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        # = start AutoML
        time_left_for_this_task = max(0, overall_limit - stop.wall_elapsed("wholething") - BUFFER)
        queue_dict[basename] = multiprocessing.Queue()
        p = multiprocessing.Process(target=start_automl.start_automl_on_dataset,
                                    args=(basename, input_dir,
                                          tmp_dataset_dir, output_dir,
                                          time_left_for_this_task, queue_dict[basename]))
        p.start()
        stop.stop_task(basename)

    # == Try to get information from all subprocesses, especially pid
    information_ready = False
    data_loading_times = dict()
    while not information_ready:
        # = We assume all jobs are running
        information_ready = True
        # = Loop over all datasets
        for basename in datanames:
            # = If this entry does not exist, the process has not yet started
            if not basename + "_ensemble" in pid_dict:
                information_ready = False
                vprint(verbose, "Waiting for run information about %s" % basename)
                try:
                    [time_needed_to_load_data, pid_smac, pid_ensembles] = queue_dict[basename].get_nowait()
                    pid_dict[basename + "_ensemble"] = pid_ensembles
                    pid_dict[basename + "_smac"] = pid_smac
                    stop.insert_task(name=basename + "_load",
                                     cpu_dur=time_needed_to_load_data,
                                     wall_dur=time_needed_to_load_data)
                except Queue.Empty:
                    continue
        vprint(verbose, "\n")
        if not information_ready:
            if stop.wall_elapsed("wholething") >= overall_limit-15:
                # = We have to stop, as there is no time left
                continue
            else:
                time.sleep(10)

    print stop

    # == And now we wait till we run out of time
    run = True
    while run:
        vprint(verbose, "#"*80)
        vprint(verbose, "# Nothing to do, %fsec left" %
               (overall_limit - stop.wall_elapsed("wholething")))
        vprint(verbose, "#"*80 + "\n")

        # == Check whether all pids are still running
        vprint(verbose, "+" + "-" * 48 + "+")
        for key in pid_dict:
            running = check_pid.check_pid(pid_dict[key])
            vprint(verbose, "|%32s|%10d|%5s|" % (key, pid_dict[key], str(running)))
        vprint(verbose, "+" + "-" * 48 + "+")

        # == List results in outputdirectory
        output_dir_list = str(os.listdir(output_dir))
        vprint(verbose, "\nOutput directory contains: %s" % output_dir_list)
        time.sleep(10)
        if stop.wall_elapsed("wholething") >= overall_limit-BUFFER_BEFORE_SENDING_SIGTERM-10:
            run = False

    # == Now it's time to terminate ... subprocesses
    # We kill processes where the cmdline() matches ones of these expressions
    smac_exp = re.compile(r"ca\.ubc\.cs\.beta\.smac\.executors\.SMACExecutor").search
    wrapper_exp = re.compile(r"wrapper\_for\_SMAC\.py$").search
    runsolver_exp = re.compile(r"runsolver$").search
    ensemble_exp = re.compile(r"ensembles\.py$").search

    stop.start_task("Shutdown")
    vprint(verbose, "Starting Shutdown, %fsec left" %
           (overall_limit-stop.wall_elapsed("wholething")))
    util.send_signal_to_our_processes(sig=15, filter=smac_exp)
    stop.stop_task("Shutdown")

    while stop.wall_elapsed("wholething") <= overall_limit - BUFFER_BEFORE_SENDING_SIGTERM + DELAY_TO_SIGKILL:
        vprint(verbose, "wait")
        time.sleep(1)

    # == Now it's time to kill ... subprocesses
    stop.start_task("Harakiri")
    vprint(verbose, "Starting Harakiri")
    util.send_signal_to_our_processes(sig=9, filter=smac_exp)
    stop.stop_task("Harakiri")

    # == Finish and show stopwatch
    stop.stop_task("submitcode")
    stop.stop_task("wholething")
    vprint(verbose, stop)

    exit(0)



