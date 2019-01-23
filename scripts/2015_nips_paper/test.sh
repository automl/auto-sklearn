#!/usr/bin/env bash
#SBATCH -p ml_cpu-ivy # partition (queue)
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -c 4 # number of cores
#SBATCH -a 1-4 # array size
#SBATCH -D /home/user/experiment # Change working_dir
#SBATCH -o log_output/out/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log_output/err/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

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
#cp -R /data/aad/openml/ $HOME/

rm -r commands.txt
rm -r log_output

# Create commands
echo "creating command file..."
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
echo "creating command file done"

# run all commands. TODO: We do this with cluster!
bash commands.txt

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
echo "merging files over tasks done"

# plotting script Usage: python plot_ranks_from_csv.py <Dataset> <model> *.csv
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

