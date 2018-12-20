#!/bin/bash

# get vanilla auto-sklaenr performance
#python score_vanilla.py --working-directory log_output --time-limit 20 --per-run-time-limit 5 --task-id 3 -s 1

# get ensemble performance.
#python score_ensemble.py --input-directory log_output -s 1 --output-file scores_ens.csv --task-id 3

dir=log_output  # working directory
test_dir=test_output
rm test_commands.txt

#python run_auto_sklearn.py --working-directory test_output --model vanilla --output-file score_vanilla.csv --task-id 58 -s 1
#python run_auto_sklearn.py --working-directory log_output --model vanilla --output-file score_vanilla.csv --task-id 3 -s 2
#python run_auto_sklearn.py --working-directory log_output --model ensemble --output-file score_ensemble.csv --task-id 3 -s 1
#python run_auto_sklearn.py --working-directory log_output --model ensemble --output-file score_ensemble.csv --task-id 3 -s 2

# we need to make a list of files to merge (for each model) to give as an argument to merge_function.
echo "creating test files..."
#files_to_merge="log_output/0/3/score_ensemble.csv log_output/1/3/score_ensemble.csv"
#test_files1="log_output/0/3/score_vanilla.csv log_output/1/3/score_vanilla.csv"
#test_files2="log_output/0/3/score_ensemble.csv log_output/1/3/score_ensemble.csv"

task_ids=$(python get_tasks.py)
seeds={1..3}

for seed in $seeds; do
    for task_id in $task_ids; do #$tasks; do
        for model in vanilla ensemble; do # metalearning meta_ensemble; do
            cmd="python run_auto_sklearn.py --working-directory $test_dir --task-id $task_id \
            -s $seed --output-file score_$model.csv --model $model"
            echo $cmd >> test_commands.txt
        done
    done
done

# run all commands. TODO: We do this with cluster!
bash test_commands.txt

# Iteratively create files to merge (for each seed)
vanilla_files=""
ensemble_files=""
for seed in $seeds; do
    for model in vanilla ensemble; do # metalearning meta_ensemble; do
        #TODO: append all file paths to the arrays. 3 should be average in the end
        vanilla_files+="$test_dir/$seed/$task_id/score_vanilla.csv "
        ensemble_files+="$test_dir/$seed/$task_id/score_ensemble.csv "
        #metalearning_filles+="$test_dir/$seed/$task_id/score_metalearning.csv "
        #meta_ensemble_filles+="$test_dir/$seed/$task_id/score_meta_ensemble.csv "
    done
done
echo "vanilla files: $vanilla_files"
echo "ensemble files: $ensemble_files"
echo "creating test files done."

# merge files (fill times, put all different seed result to one folder)
# TODO: maybe first take the average of all task runs for each seed (so for each seed there are 4 csv files)
# TODO: and then merge over seeds to create 4 csv files. How does this work with plot_ranks.py file?
echo "merging all seed runs to one file"
#python merge_test_performance_different_times.py --save merged.csv $files_to_merge
# delete this
#python merge_test_performance_different_times.py --save test_merged1.csv $vanilla_files
#python merge_test_performance_different_times.py --save test_merged2.csv $ensemble_files
echo "merging done"


# plot test performance.
# Jinu comment: This should be $model_name $csv_file $mode_name $csv_file and so on...
#python plot_test_performance_from_csv.py vanilla $test_file1 ensemble $test_file2

# plot ranks from csv.
# Usage: python plot_ranks_from_csv.py <Dataset> <model> *.csv
#python plot_ranks_from_csv.py -s plot.png tasks_s3 vanilla test_merged1.csv tasks_s3 ensemble test_merged2.csv
