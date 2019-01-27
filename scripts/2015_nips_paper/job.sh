#!/usr/bin/env bash

#bash create_commands.sh
rm -r test2_folder

python slurm_helper.py commands.txt \
    --cores 1 --timelimit 30000 --queue ml_cpu-ivy --array_min 1 --array_max 620 \
    --output test2_folder --logfiles test2_folder

