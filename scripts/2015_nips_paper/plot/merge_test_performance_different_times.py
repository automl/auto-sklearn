#!/usr/bin/env python

from argparse import ArgumentParser
import csv
import sys
import os

import numpy as np

import plot_util
import pandas as pd


def fill_trajectory(performance_list, time_list):
    # Create n series objects.
    series_list = []
    for n in range(len(time_list)):
        series_list.append(pd.Series(data=performance_list[n], index=time_list[n]))

    # Concatenate to one Series with NaN vales.
    series = pd.concat(series_list, axis=1)

    # Fill missing performance values (NaNs) with last non-NaN value.
    series = series.fillna(method='ffill')

    # Returns performance (Numpy array), time steps (list)
    return series.values, list(series.index)


def main():
    # path= ../$work_dir/vanilla/#seed/$task_id/score_vanilla.csv
    working_directory = "../log_output"

    vanilla_seed_dir = os.path.join(working_directory, 'vanilla')
    seed_list = [seed for seed in os.listdir(vanilla_seed_dir)]
    #print(seed_list)

    vanilla_task_dir = os.path.join(vanilla_seed_dir, seed_list[0])
    task_list = [task_id for task_id in os.listdir(vanilla_task_dir)]

    for model in ['vanilla', 'ensemble', 'metalearning', 'meta_ens']:
        for task_id in task_list:
            csv_files = []

            for seed in seed_list:
                # Handling the two cases separately here because they are located in different folders.
                if model in ['vanilla', 'ensemble']:
                    # no metalearning (vanilla, ensemble)
                    csv_file = os.path.join(working_directory,
                                            'vanilla',
                                            seed,
                                            task_id,
                                            "score_{}.csv".format(model)
                                            )
                    csv_files.append(csv_file)

                elif model in ['metalearning', 'meta_ens']:
                    # Metalearning (metalearning, meta_ensemble)
                    csv_file = os.path.join(working_directory,
                                            'metalearning',
                                            seed,
                                        task_id,
                                        "score_{}.csv".format(model)
                                        )
                csv_files.append(csv_file)



    # Get files and names
    arg_list = list(["dummy", ])
    arg_list.extend(unknown)
    file_list, name_list = plot_util.get_file_and_name_list(arg_list, match_file='.csv')
    del arg_list

    for time_idx in range(len(name_list)):
        print("%20s contains %d file(s)" %
              (name_list[time_idx], len(file_list[time_idx])))

    if len(file_list) > 1:
        sys.stderr.write("Cannot handle more than one experiment")
        parser.print_help()
        sys.exit(1)

    file_list = file_list[0]

    # Get data from csv
    performance_list = list()
    time_list = list()

    for fl in file_list:
        _none, csv_data = plot_util.read_csv(fl, has_header=True)
        csv_data = np.array(csv_data)
        # Replace too high values with args.maxsize
        if len(csv_data) == 0:
            print("Empty array in %s" % fl)
            continue
        if args.train:
            data = [min([args.maxvalue, float(i.strip())]) for i in
                    csv_data[:, 1]]
        else:
            data = [min([args.maxvalue, float(i.strip())]) for i in
                    csv_data[:, 2]]

        time_steps = [float(i.strip()) for i in csv_data[:, 0]]
        assert time_steps[0] == 0

        performance_list.append(data)
        time_list.append(time_steps)

    # performance : Numpy array (n_values, n_runs)
    # time_: list
    performance, time_ = fill_trajectory(
        performance_list=performance_list,
        time_list=time_list,
    )

    fh = open(args.saveTo, 'w')
    writer = csv.writer(fh)
    header = ["Time", ]
    header.extend([fl for fl in file_list])
    writer.writerow(header)
    for r, t in enumerate(time_):
        row = list([t, ])
        row.extend(["%10.5f" % performance[r, i] for i in range(len(file_list))])
        writer.writerow(row)
    fh.close()


if __name__ == "__main__":
    main()
