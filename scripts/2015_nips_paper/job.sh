#!/usr/bin/env bash

# Run created commands.
python slurm_helper.py commands.txt \
    --cores 1 --timelimit 20000 --queue ml_cpu-ivy --array_min 1 --array_max 4000 \
    --output full_run_log --logfiles full_run_log

