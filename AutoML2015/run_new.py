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
zipme = False # use this flag to enable zipping of your code submission
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
import data.data_io as data_io            # general purpose input/output functions
from data.data_io import vprint           # print only in verbose mode
from data.data_manager import DataManager # load/save data and get info about them

from util import Stopwatch, submit_process, split_data

# Define overall timelimit: 20min
OVERALL_TIMELIMIT = 60*60*20
# And buffer
BUFFER = 60

if debug_mode >= 4 or running_on_codalab: # Show library version and directory structure
    data_io.show_version()
    data_io.show_dir(run_dir)


# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:
    # Store all pid from running processes to check whether they are still alive
    pid_dict = dict()
    stop = Stopwatch.StopWatch()
    stop.start_task("wholething")
    stop.start_task("inventory")

    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2]); 
    # Move old results and create a new output directory 
    data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_io(input_dir, output_dir)
        print('\n****** Sample code version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = [] # Do not proceed with learning and testing

    stop.stop_task("inventory")
    stop.start_task("submitresults")
    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames)>0:
        vprint( verbose,  "************************************************************************")
        vprint( verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint( verbose,  "************************************************************************")
        OK = data_io.copy_results(datanames, res_dir, output_dir, verbose) # DO NOT REMOVE!
        if OK: 
            vprint( verbose,  "[+] Success")
            datanames = [] # Do not proceed with learning and testing
        else:
            vprint( verbose, "======== Some missing results on current datasets!")
            vprint( verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================
    stop.stop_task("submitresults")

    stop.start_task("submitcode")
    # ================ @CODE SUBMISSION (SUBTITUTE YOUR CODE) ================= 
    overall_time_budget = 0
    for basename in datanames:
        # Loop over datasets
        stop.start_task(basename)
        
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget.
        # Time spent to inventory data neglected.
        start = time.clock()

        stop.start_task("load_%s" % basename)
        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "======== Reading and converting data ==========")
        D = DataManager(basename, input_dir, verbose=verbose)
        print D
        X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(D.data['X_train'], D.data['Y_train'])
        del X_train, X_ensemble, Y_train
        np.save(os.path.join(D.input_dir, "true_labels_ensemble.npy"), Y_ensemble)

        # ======== Keeping track of time
        if debug_mode<1:
            time_budget = D.info['time_budget']   # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time
        overall_time_budget = overall_time_budget + time_budget
        time_spent = time.clock() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        stop.stop_task("load_%s" % basename)
        stop.start_task("start_%s" % basename)

        time_left_for_this_task = OVERALL_TIMELIMIT - \
                                  stop.wall_elapsed("wholething") - BUFFER

        pid = submit_process.run_smac(time_left_for_this_task)
        pid_dict[basename] = pid
        stop.stop_task("start_%s" % basename)

        # Start the ensemble builder
        stop.start_task("start_ensemble_builder_%s" % basename)
        ensemble_time_limit = OVERALL_TIMELIMIT - \
                              stop.wall_elapsed("wholething") - \
                              BUFFER
        submit_process.run_ensemble_builder(D.input_dir, basename, ensemble_time_limit)
        stop.stop_task("start_ensemble_builder_%s" % basename)
        stop.stop_task(basename)

    # And now we wait till the jobs are done
    print stop
    run = True
    while run:
        print "Nothing to do, wait %fsec" % (OVERALL_TIMELIMIT -
                                             stop.wall_elapsed("wholething"))
        time.sleep(10)
        if stop.wall_elapsed("wholething") >= OVERALL_TIMELIMIT:
            run = False
    print stop

    """
        # ========= Getting task and metric
        metric = D.info['metric']
        scoring_func = getattr(scores.libscores, metric)
        task_type = D.info['task']

        target_type = D.info['target_type']

        model_type = None
        multilabel = False
        multiclass = False
        sparse = False

        if task_type.lower() == 'multilabel.classification':
            multilabel = True
            model_type = AutoSklearnClassifier
        if task_type.lower() == 'regression':
            model_type = None
        if task_type.lower() == 'multiclass.classification':
            multiclass = True
            model_type = AutoSklearnClassifier
        if task_type.lower() == 'binary.classification':
            model_type = AutoSklearnClassifier

        if D.info['is_sparse'] == 1:
            sparse = True

        # ========= Creating a model, knowing its assigned task from D.info['task'].
        vprint( verbose,  "======== Creating model ==========")
        configuration_space = AutoSklearnClassifier.\
            get_hyperparameter_search_space(multiclass=multiclass,
                                            multilabel=multilabel,
                                            sparse=sparse)
        sampler = RandomSampler(configuration_space, 3)
        vprint( verbose, configuration_space)

        # ========= Iterating over learning cycles and keeping track of time
        time_spent = time.clock() - start
        vprint( verbose,  "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        time_budget = time_budget - time_spent # Remove time spent so far
        start = time.clock()              # Reset the counter
        time_spent = 0                   # Initialize time spent learning

        iteration = 0
        while time_spent <= time_budget:
            vprint( verbose,  "=========== " + basename.capitalize() +
                    " Training iteration " + str(iteration) +" ================")

            configuration = sampler.sample_configuration()
            vprint( verbose, configuration)
            model = AutoSklearnClassifier(configuration)

            model.fit(D.data['X_train'], D.data['Y_train'])
            vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.clock() - start))

            # Make predictions, take only the score of the second class
            Y_train = model.scores(D.data['X_train'])
            Y_train = np.hstack([Y_train[i][:, 1].reshape((-1, 1))
                                 for i in range(len(Y_train))])
            Y_valid = model.scores(D.data['X_valid'])
            Y_valid = np.hstack([Y_valid[i][:,1].reshape((-1, 1))
                                 for i in range(len(Y_valid))])
            Y_test = model.scores(D.data['X_test'])
            Y_test = np.hstack([Y_test[i][:, 1].reshape((-1, 1))
                                 for i in range(len(Y_test))])

            score = scoring_func(D.data['Y_train'], Y_train, task=task_type)
            vprint(verbose, score)

            vprint(verbose, "[+] Prediction success, time spent so far %5.2f sec" % (time.clock() - start))

            # Write results
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
            vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.clock() - start))
            time_spent = time.clock() - start
            vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
            iteration += 1
        """
    if zipme:
        vprint( verbose,  "========= Zipping this directory to prepare for submit ==============")
        data_io.zipdir(submission_filename + '.zip', ".")
    	
    overall_time_spent = time.clock() - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)

    stop.stop_task("submitcode")
    stop.stop_task("wholething")
    print stop

    if running_on_codalab: 
        if execution_success:
            exit(0)
        else:
            exit(1)


