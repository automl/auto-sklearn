#!/usr/bin/env bash
experiments_directory=$1
metric=$2
experiments_directory=`python -c "import os; print(os.path.abspath('$experiments_directory'))"`
commands=$experiments_directory/test_commands.txt
echo $commands

for scenario in `find $experiments_directory -name "*_${2}.scenario"`
do
    echo $scenario
    scenario_file=`python -c "import os; print(os.path.abspath('$scenario'))"`
    scenario_dir=`python -c "import os; print(os.path.dirname('$scenario_file'))"`
    trajectory_files=`find $scenario_dir  -name 'detailed-traj-run-*.csv'`
    for trajectory_file in $trajectory_files
    do
        echo smac-validate --numRun 1 --scenario $scenario_file --validate-all true --validate-only-last-incumbent false --trajectory-file $trajectory_file >> $commands
    done
done