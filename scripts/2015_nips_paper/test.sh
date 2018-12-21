#!/bin/bash

dir=log_output  # working directory
test_dir=test_output
#task_ids=$(python get_tasks.py)
task_ids="167149 167150 167152 167153"
#seeds="1 2 3 4 5 6 7 8 9 10"
seeds="1 2 3"

rm -r test_commands.txt

# Create commands
echo "creating test files..."
for seed in $seeds; do
    for task_id in $task_ids; do
        for model in vanilla ensemble metalearning; do # metalearning meta_ensemble; do
            cmd="python run_auto_sklearn.py --working-directory $test_dir --task-id $task_id \
            -s $seed --output-file score_$model.csv --model $model"
            echo $cmd >> test_commands.txt
        done
    done
done
echo "creating tesst files done"

# run all commands. TODO: We do this with cluster!
#bash test_commands.txt

# Merge over all tasks for each (model, seed) pair.

echo "merging files over tasks"
# Create strings of files to merge (over tasks)
for model in vanilla ensemble; do # metalearning meta_ensemble; do
    for seed in $seeds; do
        model_files=""
        for task_id in $task_ids; do
            model_files+="$test_dir/$seed/$task_id/score_$model.csv "
        done

        # merge them.
        python merge_test_performance_different_times.py --save $test_dir/${model}_${seed}.csv $model_files

        # average them.
        python average_test_performance_over_tasks.py --input-file $test_dir/${model}_${seed}.csv \
        --output-file $test_dir/${model}_${seed}_average.csv
    done
done
echo "merging files over tasks done"

# merge files over all seeds
echo "merging all seed runs to one file"
for model in vanilla ensemble; do # metalearning meta_ensemble; do
    model_files=""
    for seed in $seeds; do
        model_files+="$test_dir/${model}_${seed}_average.csv "
    done

    # merge them. Train argument is there to use the 2nd column as data instead of 3rd.
    python merge_test_performance_different_times.py --save $test_dir/${model}_merged.csv --train $model_files
done
echo "merging over seeds done"

# plot ranks from csv.
# Usage: python plot_ranks_from_csv.py <Dataset> <model> *.csv
python plot_ranks_from_csv.py -s plot.png tasks vanilla $test_dir/vanilla_merged.csv tasks ensemble $test_dir/ensemble_merged.csv
