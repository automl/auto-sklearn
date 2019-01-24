#!/usr/bin/env bash

dir=log_output  # working directory
#task_ids=$(python get_tasks.py)
task_ids="233 236 242 244 246 75090" #248 251 75124 253"
#seeds="1 2 3 4 5 6 7 8 9 10"
seeds="0 1 2 3 4"
time_limit=100
per_runtime_limit=33

# If running on cluster, first copy the cached openml dataset to the home directory.
# This is necessary because there are file handling operations which
# seem to require root permission. Skip this if not running on cluster.
#TODO: do this to current working directory instead of home!
#cp -R /data/aad/openml/ $HOME/

rm -r commands.txt
rm -r log_output

# Create commands
echo "creating commands.txt file..."
for seed in $seeds; do
    for task_id in $task_ids; do
        for model in vanilla ensemble metalearning meta_ensemble; do
            # put vanilla and ensemble in one file and meta and meta_ens in another.
            if [ "$model" == "ensemble" ]
            then
                cmd="python run_auto_sklearn.py --working-directory $dir/vanilla --task-id $task_id \
                -s $seed --output-file score_$model.csv --model $model --time-limit $time_limit \
                --per-runtime-limit $per_runtime_limit "
            elif [ "$model" == "meta_ensemble" ]
            then
                cmd="python run_auto_sklearn.py --working-directory $dir/metalearning --task-id $task_id \
                -s $seed --output-file score_$model.csv --model $model --time-limit $time_limit \
                --per-runtime-limit $per_runtime_limit "
            else
                cmd="python run_auto_sklearn.py --working-directory $dir/$model --task-id $task_id \
                -s $seed --output-file score_$model.csv --model $model --time-limit $time_limit \
                --per-runtime-limit $per_runtime_limit "
            fi
            echo $cmd >> commands.txt
        done
    done
done
echo "creating commands.txt done"
