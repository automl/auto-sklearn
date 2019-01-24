#!/usr/bin/env bash

bash create_commands.sh
rm -r test_folder

python slurm_helper.py commands.txt \
    --cores 5 --timelimit 7200 --queue mlstudents_cpu-ivy --array_min 1 --array_max 20 \
    --output test_folder --logfiles test_folder

