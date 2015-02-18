from argparse import ArgumentParser
from collections import defaultdict
import cPickle
from itertools import product
import os
import re

from openml.apiconnector import APIConnector

from joblib import Memory
import matplotlib.pyplot as plt
import numpy as np


def find_pickle_files(directory):
    # Cannot use defaultdicts of defaultdicts here, because I couldn't pickle
    #  them
    pickle_files = dict()
    for metric in ["bac_metric", "auc_metric", "f1_metric", "pac_metric"]:
        pickle_files[metric] = defaultdict(list)

    if not directory.endswith("/"):
        directory += "/"

    metric_re = re.compile(r"[a-zA-Z0-9]{1,3}_metric")
    did_re = re.compile(r"-?[0-9]{1,5}")

    for root, dirs, files in os.walk(directory):
        for f in files:
            if "smac_2_08_00-master.pkl" in f.lower() and \
                    not "pkl.lock" in f.lower():
                full_name = os.path.join(root, f)
                relative_name = full_name.replace(directory, "")

                res = metric_re.search(relative_name)
                metric = res.group()

                res = did_re.search(relative_name)
                did = int(float(res.group()))

                pickle_files[metric][did].append(full_name)

    return pickle_files


def read_runs_from_pickle_files(pickle_files):
    runs = []
    for pkl_file in pickle_files:
        with open(pkl_file) as fh:
            trials = cPickle.load(fh)
            for trial in trials['trials']:
                params = trial['params']
                for key in params:
                    try:
                        params[key] = float(params[key])
                    except:
                        pass

                result = np.nanmean(trial['instance_results'])
                runtime = np.nanmean(trial['instance_durations'])

                runs.append((params['classifier'],
                             params['preprocessor'],
                             result,
                             runtime))

    return runs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="CSV file describing for which "
                                      "datasets to create plot.")
    parser.add_argument("experiment_directory",
                        help="Directory which contains the experiments to "
                             "plot.")
    args = parser.parse_args()

    plot_dir = os.path.join(args.experiment_directory, "plots")
    try:
        os.mkdir(plot_dir)
    except:
        pass

    dataset_ids = []
    with open(args.input) as fh:
        for row in fh:
            dataset_ids.append(int(float(row.strip())))

    api = APIConnector(authenticate=False)

    mem = Memory(cachedir="/tmp/joblib")

    find_pickle_files_cached = mem.cache(find_pickle_files)
    pickle_files = find_pickle_files_cached(args.experiment_directory)
    metrics = ["bac_metric", "auc_metric",
               "f1_metric", "pac_metric"]
    metrics = ["bac_metric"]

    for metric, did in product(metrics, dataset_ids):
        if pickle_files.get(metric).get(did) is None:
            continue

        dataset = api.get_cached_dataset(did)
        X, y, categorical = dataset.get_pandas(target=dataset.default_target_attribute)
        subtitle = "%s (%dx%d)" % (dataset.name, X.shape[0], X.shape[1])

        read_runs_from_pickle_files_cached = mem.cache(read_runs_from_pickle_files)
        runs = read_runs_from_pickle_files(pickle_files.get(metric).get(did))

        results_per_preclf = defaultdict(list)
        runtimes_per_preclf = defaultdict(list)
        for run in runs:
            results_per_preclf[run[0] + " " + run[1]].append(run[2])
            runtimes_per_preclf[run[0] + " " + run[1]].append(run[3])

        bp = plt.boxplot([results_per_preclf[key] for key in
                          sorted(results_per_preclf)], whis="range")
        plt.xticks(range(1, 1+len(results_per_preclf)),
            [key for key in sorted(results_per_preclf)], rotation=90)
        plt.plot()
        plt.tick_params(labelsize=6)
        plt.title(subtitle, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "%s_%d.png" % (metric, did)),
                    figsize=(12,8), dpi=200)
        plt.clf()

        bp = plt.boxplot([runtimes_per_preclf[key] for key in
                          sorted(runtimes_per_preclf)], whis="range")
        plt.xticks(range(1, 1 + len(runtimes_per_preclf)),
                   [key for key in sorted(runtimes_per_preclf)], rotation=90)
        plt.plot()
        plt.tick_params(labelsize=6)
        plt.title(subtitle, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "%s_%d.png" % ("runtime", did)),
                    figsize=(12, 8), dpi=200)
        plt.clf()