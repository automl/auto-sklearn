#!/bin/bash

# get vanilla auto-sklaenr performance
#python score_vanilla.py --working-directory log_output --time-limit 20 --per-run-time-limit 5 --task-id 3 -s 1

# get ensemble performance.
#python score_ensemble.py --input-directory log_output -s 1 --output-file scores_ens.csv --task-id 3

#python run_auto_sklearn.py --working-directory log_output --model vanilla --output-file score_vanilla.csv --task-id 3 -s 0
#python run_auto_sklearn.py --working-directory log_output --model vanilla --output-file score_vanilla.csv --task-id 3 -s 1
#python run_auto_sklearn.py --working-directory log_output --model ensemble --output-file score_ensemble.csv --task-id 3 -s 0
#python run_auto_sklearn.py --working-directory log_output --model ensemble --output-file score_ensemble.csv --task-id 3 -s 1

# we need to make a list of files to merge (for each model) to give as an argument to merge_function.
files_to_merge="log_output/0/3/score_ensemble.csv log_output/1/3/score_ensemble.csv"

# merge files (fill times)
python merge_test_performance_different_times.py --save merged.csv $files_to_merge

# plot ranks from csv
#python plot_ranks_from_csv.py Task3 ensemble -s plot.png $files_to_merge

rm commands.txt

dir=log_output  # working directory
tasks=$(python get_tasks.py)
echo $tasks


for seed in {1..10}; do
    for task in {1..5}; do #$tasks; do
        for model in vanilla ensemble; do # metalearning meta_ensemble; do
            cmd="python run_auto_sklearn.py --working-directory $dir --task-id $task \
            -s $seed --output-file score_$model --model $model"
            echo $cmd >> commands.txt
        done
    done
done


