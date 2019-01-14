#!/bin/bash

dir=log_output  # working directory
test_dir=test_output
#task_ids=$(python get_tasks.py)
task_ids="233 236 242 244 246 75090 248 251 75124 253"
#seeds="1 2 3 4 5 6 7 8 9 10"
seeds="1"
time_limit=20
per_runtime_limit=5

rm -r test_commands.txt

# Create commands
echo "creating test files..."
for seed in $seeds; do
    for task_id in $task_ids; do
        for model in vanilla; do #vanilla ensemble metalearning; do # meta_ensemble; do
            cmd="python run_auto_sklearn.py --working-directory $test_dir --task-id $task_id \
            -s $seed --output-file score_$model.csv --model $model --time-limit $time_limit \
            --per-runtime-limit $per_runtime_limit "
            echo $cmd >> test_commands.txt
        done
    done
done
echo "creating tesst files done"

# run all commands. TODO: We do this with cluster!
bash test_commands.txt

echo "merging files over seeds for each tasks"
# Create strings of files to merge (over seeds)
for model in vanilla ensemble; do # metalearning meta_ensemble; do
    for task_id in $task_ids; do
        model_files=""
        for seed in $seeds; do
            model_files+="$test_dir/$seed/$task_id/score_$model.csv "
        done

        # merge them.
        #python merge_test_performance_different_times.py --save $test_dir/${model}_${task_id}.csv $model_files
    done
done
echo "merging files over tasks done"

# plotting script Usage: python plot_ranks_from_csv.py <Dataset> <model> *.csv
# create command for plot_ranks_from_csv.py. Iteratively append <Dataset> <model> *.csv to
# cmd for each (task_id, model, file) pairs.
cmd="python plot_ranks_from_csv.py -s plot.png "
for model in vanilla ensemble; do # metalearning meta_ensemble; do
    for task_id in $task_ids; do
        cmd+="$task_id $model $test_dir/${model}_${task_id}.csv " #TODO: change test dir properly
    done
done
# plot ranks from csv.
eval $cmd

