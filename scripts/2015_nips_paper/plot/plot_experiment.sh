#!/usr/bin/env bash
##SBATCH -p ml_cpu-ivy
#SBATCH --mem 4000
#SBATCH -t 0-05:00
#SBATCH -c 1
#SBATCH -o plot.out # STDOUT
#SBATCH -e plot.err # STDERR

dir=log_output  # working directory
#task_ids=$(python get_tasks.py)
task_ids="233 236 242 244 246 248 251 252 253 254 256 258 260 261 262 266 273 275 288 2117 2118 2119 2120 2122 2123 2350 3043 3044 75090 75092 75093 75098 75099 75100 75103 75104 75105 75106 75107 75108 75111 75112 75113 75114 75115 75116 75117 75119 75120 75121 75122 75125 75126 75129 75131 75133 75136 75137 75138 75139 75140 75142 75143 75146 75147 75148 75149 75150 75151 75152 75153 75155 75157 75159 75160 75161 75162 75163 75164 75165 75166 75168 75169 75170 75171 75172 75173 75174 75175 75176 75179 75180 75182 75183 75184 75185 75186 75188 75189 75190 75191 75192 75194 75195 75196 75197 75198 75199 75200 75201 75202 75203 75204 75205 75206 75207 75208 75209 75210 75212 75213 75216 75218 75220 75222 75224 75226 75228 75229 75233 75238 75240 75244 75245 75246 75247 75248 75249 75251 "

seeds="0 1 2 3 4 5 6 7 8 9 "
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
cmd="$cmd --samples $n_samples "
eval $cmd

