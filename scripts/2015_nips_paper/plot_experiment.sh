#!/usr/bin/env bash

dir=log_output  # working directory
#task_ids=$(python get_tasks.py)
task_ids="233 236 242 244 246 75090" #248 251 75124 253"
#seeds="1 2 3 4 5 6 7 8 9 10"
seeds="0 1 2 3 4"
time_limit=100
per_runtime_limit=33

echo "merging files over seeds for each tasks"
# Create strings of files to merge (over seeds)
for model in vanilla ensemble metalearning meta_ensemble; do
    for task_id in $task_ids; do
        model_files=""
        for seed in $seeds; do
            if [ "$model" == "ensemble" ]
            then
                model_files+="$dir/vanilla/$seed/$task_id/score_$model.csv "
            elif [ "$model" == "meta_ensemble" ]
            then
                model_files+="$dir/metalearning/$seed/$task_id/score_$model.csv "
            else
                model_files+="$dir/$model/$seed/$task_id/score_$model.csv "
            fi

        done

        # merge them.
        echo $model_files
        python merge_test_performance_different_times.py --save $dir/${model}_${task_id}.csv $model_files
    done
done
echo "merging files over seeds done"

# create command for plot_ranks_from_csv.py. Iteratively append <Dataset> <model> *.csv to
# cmd for each (task_id, model, file) pairs.
cmd="python plot_ranks_from_csv.py -s plot.png "
for model in vanilla ensemble metalearning meta_ensemble; do
    for task_id in $task_ids; do
        cmd="$cmd $task_id $model $dir/${model}_${task_id}.csv"
    done
done
# plot ranks from csv.
eval $cmd

