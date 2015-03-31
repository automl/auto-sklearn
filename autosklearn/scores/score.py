#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
from argparse import ArgumentParser
import os

from sys import argv
import yaml
from autosklearn.scores.libscores import * # Library of scores we implemented

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 1

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 0.8
    
# =============================== MAIN ========================================
    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_directory", help="Directory in which the "
                                                  "dataset resides.")
    parser.add_argument("prediction_directory", help="Directory in which the "
                                                     "predictions reside.")
    parser.add_argument("output_directory", help="Directory where the scores "
                                                 "will be written.")
    args = parser.parse_args()

    #### INPUT/OUTPUT: Get input and output directory name
    input_dir = args.dataset_directory
    prediction_dir = args.prediction_directory
    output_dir = args.output_directory

    # Create the output directory, if it does not already exist and open output files  
    mkdir(output_dir) 
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'wb')
    
    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(input_dir, '*.solution')))

    # Loop over files in solution directory and search for predictions with extension .predict having the same basename
    set_num = 1
    for solution_file in solution_names:
        # Extract the dataset name from the file name
        basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.')-1]
        basename, seperator, set_name = basename.rpartition("_")

        # Load the info file and get the task and metric
        info_file = ls(os.path.join(input_dir, basename + '*_public.info'))[0]
        info = get_info (info_file)    
        score_name = info['task'][0:-15] + info['metric'][0:-7].upper()
        predict_name = basename
        try:
            # Get the last prediction from the res subdirectory (must end with '.predict')
            predict_file = ls(os.path.join(prediction_dir, basename + "_" +
                                           set_name + '*.predict'))[-1]
            if (predict_file == []): raise IOError('Missing prediction file {}'.format(basename))
            predict_name = predict_file[-predict_file[::-1].index(filesep):-predict_file[::-1].index('.')-1]
            # Read the solution and prediction values into numpy arrays
            solution = read_array(solution_file)
            prediction = read_array(predict_file)
            if(solution.shape!=prediction.shape): raise ValueError("Bad prediction shape {}".format(prediction.shape))

            try:
                # Compute the score prescribed by the info file (for regression scores, no normalization)            
                if info['metric']=='r2_metric' or info['metric']=='a_metric': 
                    # Remove NaN and Inf for regression
                    solution = sanitize_array (solution); prediction = sanitize_array (prediction)  
                    score = eval(info['metric'] + '(solution, prediction, "' + info['task'] + '")')
                else:
                    # Compute version that is normalized (for classification scores). This does nothing if all values are already in [0, 1]
                    [csolution, cprediction] = normalize_array (solution, prediction)
                    score = eval(info['metric'] + '(csolution, cprediction, "' + info['task'] + '")')
                print("======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=%0.12f =======" % score)                
                html_file.write("======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=%0.12f =======\n" % score)
            except:
                raise Exception('Error in calculation of the specific score of the task')
                
            if debug_mode>0: 
                scores = compute_all_scores(solution, prediction)
                write_scores(html_file, scores)
                    
        except Exception as inst:
            score = missing_score
            print("======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=ERROR =======")
            html_file.write("======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=ERROR =======\n")            
            print inst
            
        # Write score corresponding to selected task and metric to the output file
        score_file.write("set%d" % set_num + "_score: %0.12f\n" % score)
        set_num=set_num+1            
    # End loop for solution_file in solution_names

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])
    except:
        score_file.write("Duration: 0\n")
		
    html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode>1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(input_dir, output_dir)
        show_version(scoring_version)
		
    #exit(0)



