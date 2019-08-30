import os

import arff
from shutil import copyfile


def remove_dataset_from_aslib_arff(input_file,
                                   output_file,
                                   id,
                                   ):
    with open(input_file) as fh:
        arff_object = arff.load(fh)
    for i in range(len(arff_object['data']) - 1, -1, -1):
        if str(arff_object['data'][i][0]) == str(id):
            del arff_object['data'][i]

    with open(output_file, "w") as fh:
        arff.dump(arff_object, fh)
    del arff_object


def remove_dataset(metadata_directory,
                   output_directory,
                   id,
                   ):
    metadata_sub_directories = os.listdir(metadata_directory)

    for metadata_sub_directory in metadata_sub_directories:
        subdir = os.path.join(metadata_directory, metadata_sub_directory)
        output_subdir = os.path.join(output_directory, metadata_sub_directory)
        try:
            os.makedirs(output_subdir)
        except OSError:
            pass

        arf = "algorithm_runs.arff"
        algorithm_runs_file = os.path.join(subdir, arf)
        output_file = os.path.join(output_subdir, arf)
        remove_dataset_from_aslib_arff(algorithm_runs_file, output_file, id)

        fcf = "feature_costs.arff"
        feature_costs_file = os.path.join(subdir, fcf)
        output_file = os.path.join(output_subdir, fcf)
        remove_dataset_from_aslib_arff(feature_costs_file, output_file, id)

        frf = "feature_runstatus.arff"
        feature_runstatus_file = os.path.join(subdir, frf)
        output_file = os.path.join(output_subdir, frf)
        remove_dataset_from_aslib_arff(feature_runstatus_file, output_file, id)

        fvf = "feature_values.arff"
        features_values_file = os.path.join(subdir, fvf)
        output_file = os.path.join(output_subdir, fvf)
        remove_dataset_from_aslib_arff(features_values_file, output_file, id)

        desc = "description.txt"
        description_file = os.path.join(subdir, desc)
        output_file = os.path.join(output_subdir, desc)
        copyfile(description_file, output_file)

        configs = "configurations.csv"
        configs_file = os.path.join(subdir, configs)
        output_file = os.path.join(output_subdir, configs)
        copyfile(configs_file, output_file)
