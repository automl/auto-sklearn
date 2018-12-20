#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import csv
import scipy.stats
import sys

import numpy as np
import itertools

import plot_util
import merge_test_performance_different_times as merge
import plot_test_performance_from_csv


def calculate_ranking(performances, estimators, bootstrap_samples=500):
    num_steps = len(performances[estimators[0]]["performances"][0])
    num_estimators = len(estimators)
    ranking = np.zeros((num_steps, len(estimators)), dtype=np.float64)

    rs = np.random.RandomState(1)

    combinations = []
    maximum = [len(performances[name]) for name in estimators]
    for j in range(bootstrap_samples):
        combination = []
        for idx in range(num_estimators):
            combination.append(rs.randint(maximum[idx]))
        combinations.append(np.array(combination))

    # Initializes ranking array
    # Not sure whether we need this
    #for j, est in enumerate(estimators):
    #    ranking[0][j] = np.mean(range(1, len(estimators) + 1))

    for i in range(ranking.shape[0]):
        num_products = 0

        for combination in combinations:
            ranks = scipy.stats.rankdata(
                [np.round(
                    performances[estimators[idx]]["performances"][number][i], 5)
                 for idx, number in enumerate(combination)])
            num_products += 1
            for j, est in enumerate(estimators):
                ranking[i][j] += ranks[j]

        for j, est in enumerate(estimators):
            ranking[i][j] = ranking[i][j] / num_products

    return list(np.transpose(ranking)), estimators


def main():
    prog = "python plot_ranks_from_csv.py <Dataset> <model> " \
           "*.csv ... "
    description = "Plot ranks over different datasets"

    parser = ArgumentParser(description=description, prog=prog,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # General Options
    parser.add_argument("--logy", action="store_true", dest="logy",
                        default=False, help="Plot y-axis on log scale")
    parser.add_argument("--logx", action="store_true", dest="logx",
                        default=False, help="Plot x-axis on log scale")
    parser.add_argument("--ymax", dest="ymax", type=float,
                        default=None, help="Maximum of the y-axis")
    parser.add_argument("--ymin", dest="ymin", type=float,
                        default=None, help="Minimum of the y-axis")
    parser.add_argument("--xmax", dest="xmax", type=float,
                        default=None, help="Maximum of the x-axis")
    parser.add_argument("--xmin", dest="xmin", type=float,
                        default=None, help="Minimum of the x-axis")
    parser.add_argument("-s", "--save", dest="save",
                        default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--maxvalue", dest="maxvalue", type=float,
                        default=sys.maxsize,
                        help="Replace all values higher than this?")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        default=False, help="print number of runs on plot")
    parser.add_argument("--samples", dest="samples", type=int,
                        default=1000, help="Number of bootstrap samples to plot")
    parser.add_argument("--figsize", nargs=2, type=float,
                        help="Set matplotlib argument figsize.")
    parser.add_argument("--sort", action="store_true",
                        help="Sort the legend.")

    # Properties
    # We need this to show defaults for -h
    defaults = plot_util.get_defaults()
    for key in defaults:
        parser.add_argument("--%s" % key, dest=key, default=None,
                            help="%s, default: %s" % (key, str(defaults[key])))

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    if len(unknown) < 2:
        print("To less arguments given")
        parser.print_help()
        sys.exit(1)

    # Get files and names
    file_list, name_list = plot_util.get_file_and_name_list(unknown,
                                                            match_file='.csv',
                                                            len_name=2)

    for idx in range(len(name_list)):
        assert len(file_list[idx]) == 1, "%s: %s" % (name_list[idx],
                                                     file_list[idx])
        print("%20s contains %d file(s)" % (name_list[idx], len(file_list[idx])))

    dataset_dict = OrderedDict()
    estimator_list = list()
    dataset_list = list()
    print(name_list)
    for idx, desc in enumerate(name_list):
        dataset = desc[0]
        est = desc[1]
        if est not in estimator_list:
            estimator_list.append(est)
        if dataset not in dataset_list:
            dataset_list.append(dataset)
        if dataset not in dataset_dict:
            dataset_dict[dataset] = OrderedDict()
        t = None
        p = None
        print("Processing %s, %s" % (dataset, est))
        fh = open(file_list[idx][0], 'r')
        reader = csv.reader(fh)
        for row in reader:
            if t is None:
                # first row
                p = list([list() for i in range(len(row)-1)])
                t = list()
                dataset_dict[dataset][est] = OrderedDict()
                continue
            t.append(float(row[0]))
            del row[0]
            [p[i].append(float(row[i])) for i in range(len(row))]
        dataset_dict[dataset][est]["times"] = [t for i in range(len(p))]
        dataset_dict[dataset][est]["performances"] = p

    # Make lists
    if args.sort:
        estimator_list = sorted(list(estimator_list))
    else:
        estimator_list = list(estimator_list)
    dataset_list = list(dataset_list)

    print("Found datasets: %s" % str(dataset_list))
    print("Found estimators: %s" % str(estimator_list))

    for dataset in dataset_list:
        print("Processing dataset: %s" % dataset)
        if dataset not in dataset_dict:
            # This should never happen
            raise ValueError("Dataset %s lost" % dataset)

        # We have a list of lists of lists, but we need a list of lists
        tmp_p_list = list()
        tmp_t_list = list()
        len_list = list()       # holds num of arrays for each est
        for est in estimator_list:
            # put all performances in one list = flatten
            if est not in dataset_dict[dataset]:
                raise ValueError("Estimator %s is not given for dataset %s" %
                                 (est, dataset))

            len_list.append(len(dataset_dict[dataset][est]["performances"]))
            tmp_p_list.extend(dataset_dict[dataset][est]["performances"])
            tmp_t_list.extend(dataset_dict[dataset][est]["times"])

        # sanity check
        assert len(tmp_t_list) == len(tmp_p_list)
        assert len(tmp_t_list[0]) == len(tmp_p_list[0])
        p, t = merge.fill_trajectory(performance_list=tmp_p_list,
                                     time_list=tmp_t_list)

        # Now we can refill the dict using len_list as it tells us
        # which arrays belong to which estimator
        for idx, est in enumerate(estimator_list):
            dataset_dict[dataset][est]['performances'] = p[:len_list[idx]]
            # sanity check
            assert len(dataset_dict[dataset][est]['performances'][0]) == len(t)
            del p[:len_list[idx]]
        dataset_dict[dataset]['time'] = t

    # Calculate rankings
    ranking_list = list()
    time_list = list()
    for dataset in dataset_list:
        ranking, e_list = calculate_ranking(performances=dataset_dict[dataset],
                                            estimators=estimator_list, bootstrap_samples=args.samples)
        ranking_list.extend(ranking)
        assert len(e_list) == len(estimator_list)
        time_list.extend([dataset_dict[dataset]["time"] for i in range(len(e_list))])

    # Fill trajectories as ranks are calculated on different time steps
    # sanity check
    assert len(ranking_list) == len(time_list)
    assert len(ranking_list[0]) == len(time_list[0]), "%d is not %d" % \
                                                      (len(ranking_list[0]),
                                                       len(time_list[0]))
    p, times = merge.fill_trajectory(performance_list=ranking_list,
                                     time_list=time_list)
    del ranking_list, dataset_dict

    performance_list = [list() for e in estimator_list]
    time_list = [times for e in estimator_list]
    for idd, dataset in enumerate(dataset_list):
        for ide, est in enumerate(estimator_list):
            performance_list[ide].append(p[idd*(len(estimator_list))+ide])


    prop = {}
    args_dict = vars(args)
    for key in defaults:
        prop[key] = args_dict[key]
    #prop['linestyles'] = itertools.cycle(["-", ":"])


    plot_test_performance_from_csv.\
        plot_optimization_trace(time_list=time_list,
                                performance_list=performance_list,
                                title=args.title, name_list=estimator_list,
                                logy=args.logy, logx=args.logx, save=args.save,
                                y_min=args.ymin, y_max=args.ymax,
                                x_min=args.xmin, x_max=args.xmax,
                                ylabel="average rank",
                                scale_std=0, properties=prop,
                                figsize=tuple(args.figsize) if args.figsize
                                        else None)

if __name__ == "__main__":
    main()