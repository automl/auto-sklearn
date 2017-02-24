# How to update metadata

(to be moved to the documentation)

## 1. Create a working directory and set the task type

The working directory will be used to save all temporary and final output.

    working_directory=~/auto-sklearn-metadata/001
    mkdir -p $working_directory

The task type defines whether you want update classification or regression
metadata:

    task_type=classification

or

    task_type=regression

## 2. Install the OpenML package and create an OpenML account

Read the [OpenML python package manual](https://openml.github.io/openml-python) for this.

## 3. Create configuration commands

    python3 01_create_commands.py --working-directory $working_directory --task-type $task_type

This will create a file with all commands necessary to run auto-sklearn on a
large number of datasets from OpenML. You can change the task IDs or the way
how the datasets are loaded in the file `update_metadata_util.py`. To change
the time used for configuration, you can alter the commands file which will
be written to disk.

## 4. Run all configuration runs

On hardware of your choice. It is recommended to run all runs in parallel in
order to get the results in a reasonable amount of time.

## 5. Get the test performance of these configurations

    python3 02_retrieve_metadata.py --working-directory $working_directory --task-type $task_type

## 6. Calculate metafeatures

    python3 03_calculate_metafeatures.py --working-directory $working_directory --task-type $task_type

## 7. Create aslib files

    python3 04_create_aslib_files.py --working-directory $working_directory --task-type $task_type
