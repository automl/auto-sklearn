#!/usr/bin/env bash
##SBATCH -p ml_cpu-ivy # partition (queue)
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -o plot.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e plot.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

dir=log_output  # working directory
#task_ids=$(python get_tasks.py)
#task_ids="233 236 242 244 246 248 251 252 253 254 258 260 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 75090 75092 75093 75098 75099 75100 75103 75104 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75140 75142 75143 "
#task_ids="233 236 242 244 246 251 252 253 254 258 260 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 75090 75092 75093 75098 75100 75103 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75140 75142 75143 "

# This is what I used for last experiment.
#task_ids="233 236 242 244 246 248 251 252 253 254 258 260 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 75090 75092 75093 75098 75099 75100 75103 75104 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75140 75142 75143 75146 75147 75148 75149 75150 75151 75152 75153 75155 75157 "

# Exclude 75140 cause it is not ending somehow
#task_ids="233 236 242 244 246 248 251 252 253 254 258 260 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 75090 75092 75093 75098 75099 75100 75103 75104 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75142 75143 75146 75147 75148 75149 75150 75151 75152 75153 75155 75157 "

# Exclude runs which did not work
task_ids="233 236 242 244 246 248 251 252 253 254 258 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 75090 75092 75093 75098 75099 75100 75103 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75142 75143 75146 75147 75149 75150 75151 75152 75153 75155 75157 "
#seeds="1 2 3 4 5 6 7 8 9 10"
seeds="0 1 2 3 4 5 "
# number of bootstrap samples to use
n_samples=500

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
        #python merge_test_performance_different_times.py --save $dir/${model}_${task_id}.csv $model_files
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
cmd="$cmd --samples $n_samples "
eval $cmd

