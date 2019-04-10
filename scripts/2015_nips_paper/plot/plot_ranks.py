#!/usr/bin/env python3

import csv
import sys
import os

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def read_csv(fn, has_header=True, data_type=str):
    """
    Function which reads the csv files containing trajectories
    of the auto-sklearn runs.
    """
    data = list()
    header = None
    with open(fn, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if header is None and has_header:
                header = row
                continue
            data.append(list(map(data_type, [i.strip() for i in row])))
    return header, data


def fill_trajectory(performance_list, time_list):
    # Create n series objects.
    series_list = []
    for n in range(len(time_list)):
        series_list.append(pd.Series(data=performance_list[n], index=time_list[n]))

    # Concatenate to one Series with NaN vales.
    series = pd.concat(series_list, axis=1)

    # Fill missing performance values (NaNs) with last non-NaN value.
    series = series.fillna(method='ffill')

    # return the trajectories over seeds (series object)
    return series


def main():
    # name of the file where the plot is stored
    saveto = "../plot.png"
    # runtime of each experiment
    max_runtime = 3600
    # folder where all trajectories are stored.
    working_directory = "../log_output"

    # list of models
    model_list = ['vanilla', 'ensemble', 'metalearning', 'meta_ensemble']

    # list of seeds
    seed_dir = os.path.join(working_directory, 'vanilla')
    seed_list = [seed for seed in os.listdir(seed_dir)]

    # list of tasks
    vanilla_task_dir = os.path.join(seed_dir, seed_list[0])
    task_list = [task_id for task_id in os.listdir(vanilla_task_dir)]

    # Step 1. Merge all trajectories into one Dataframe object.
    #####################################################################################
    all_trajectories = []

    for model in model_list:
        trajectories = []
        for task_id in task_list:
            csv_files = []

            for seed in seed_list:
                # collect all csv files of different seeds for current model and
                # current task.
                if model in ['vanilla', 'ensemble']:
                    csv_file = os.path.join(working_directory,
                                            'vanilla',
                                            seed,
                                            task_id,
                                            "score_{}.csv".format(model)
                                            )

                elif model in ['metalearning', 'meta_ensemble']:
                    csv_file = os.path.join(working_directory,
                                            'metalearning',
                                            seed,
                                            task_id,
                                            "score_{}.csv".format(model),
                                            )
                csv_files.append(csv_file)

            performance_list = []
            time_list = []

            # Get data from csv
            for fl in csv_files:
                _, csv_data = read_csv(fl, has_header=True)
                csv_data = np.array(csv_data)
                # Replace too high values with args.maxsize
                data = [min([sys.maxsize, float(i.strip())]) for i in
                        csv_data[:, 2]]  # test trajectories are stored in third column

                time_steps = [float(i.strip()) for i in csv_data[:, 0]]
                assert time_steps[0] == 0

                performance_list.append(data)
                time_list.append(time_steps)

            # trajectory is the pd.Series object containing all seed runs of the
            # current model and current task.
            trajectory = fill_trajectory(performance_list, time_list)
            trajectories.append(trajectory)

        # list[list[pd.Series]]
        all_trajectories.append(trajectories)

    # Step 2. Compute average ranks of the trajectories.
    #####################################################################################
    all_rankings = []
    n_iter = 500  # number of bootstrap samples to use for estimating the ranks.
    n_tasks = len(task_list)

    for i in range(n_iter):
        pick = np.random.choice(all_trajectories[0][0].shape[1],
                                size=(len(model_list)))

        for j in range(n_tasks):
            all_trajectories_tmp = pd.DataFrame(
                {model_list[k]: at[j].iloc[:, pick[k]] for
                 k, at in enumerate(all_trajectories)}
            )
            all_trajectories_tmp = all_trajectories_tmp.fillna(method='ffill', axis=0)
            r_tmp = all_trajectories_tmp.rank(axis=1)
            all_rankings.append(r_tmp)

    final_ranks = []
    for i, model in enumerate(model_list):
        ranks_for_model = []
        for ranking in all_rankings:
            ranks_for_model.append(ranking.loc[:, model])
        ranks_for_model = pd.DataFrame(ranks_for_model)
        ranks_for_model = ranks_for_model.fillna(method='ffill', axis=1)
        final_ranks.append(ranks_for_model.mean(skipna=True))

    # Step 3. Plot the average ranks over time.
    #####################################################################################
    for i, model in enumerate(model_list):
        X_data = []
        y_data = []
        for x, y in final_ranks[i].iteritems():
            X_data.append(x)
            y_data.append(y)
        X_data.append(max_runtime)
        y_data.append(y)
        plt.plot(X_data, y_data, label=model)
        plt.xlabel('time [sec]')
        plt.ylabel('average rank')
        plt.legend()
    plt.savefig(saveto)


if __name__ == "__main__":
    main()
