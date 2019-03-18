#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import itertools
import sys

import matplotlib
from matplotlib.pyplot import tight_layout, figure, subplot, savefig, show
import matplotlib.gridspec
import numpy as np
import plot_util

matplotlib.pyplot.switch_backend('agg')  # For error invalid DISPLAY variable
matplotlib.rcParams['pdf.fonttype'] = 42


def plot_optimization_trace(time_list, performance_list, name_list, title=None,
                            logy=False, logx=False, save="", properties=None,
                            y_min=None, y_max=None, x_min=None, x_max=None,
                            ylabel="Performance", scale_std=1,
                            scale_mean=1, figsize=None, aggregation='mean'):
    # complete properties
    if properties is None:
        properties = dict()
    properties['markers'] = itertools.cycle(['o', 's', '^', '*'])
    properties = plot_util.fill_with_defaults(properties)

    # Set up figure
    ratio = 5
    if figsize is None:
        figsize = (8, 6)
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=int(properties['dpi']), figsize=figsize)
    ax1 = subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color=properties["gridcolor"],
             alpha=float(properties["gridalpha"]))

    if title is not None:
        fig.suptitle(title, fontsize=int(properties["titlefontsize"]))

    auto_y_min = sys.maxsize
    auto_y_max = -sys.maxsize
    auto_x_min = sys.maxsize
    auto_x_max = -sys.maxsize

    for idx, performance in enumerate(performance_list):
        color = next(properties["colors"])
        marker = next(properties["markers"])
        linestyle = next(properties["linestyles"])

        replaces = {'autosklearn': 'auto-sklearn',
                    'adaboost': 'AdaBoost',
                    'bernoulli_nb': u'Bernoulli naïve bayes',
                    'extra_trees': 'extreml. rand. trees',
                    'gaussian_nb': u'Gaussian naïve bayes',
                    'k_nearest_neighbors': 'kNN',
                    'lda': 'LDA',
                    'liblinear_svc': 'linear SVM',
                    'libsvm_svc': 'kernel SVM',
                    'multinomial_nb': u'multinomial naïve bayes',
                    'qda': 'QDA',
                    'sgd': 'SGD',
                    'extra_trees_preproc_for_classification':
                        'extreml. rand. trees prepr.',
                    'fast_ica': 'fast ICA',
                    'kernel_pca': 'kernel PCA',
                    'kitchen_sinks': 'rand. kitchen sinks',
                    'liblinear_svc_preprocessor': 'linear SVM preproc.',
                    'pca': 'PCA',
                    'random_trees_embedding': 'random trees embed.'}
        for key, value in replaces.items():
            name_list[idx] = name_list[idx].replace(key, value)

        name_list[idx] = name_list[idx].replace("_", " ")

        if logy:
            performance = np.log10(performance)
        if logx and time_list[idx][0] == 0:
            time_list[idx][0] = 10**-1

        performance = np.array(performance) * scale_mean

        if aggregation == 'median':
            line = np.median(performance, axis=0)
            upper = np.percentile(performance, q=95, axis=0)
            lower = np.percentile(performance, q=5, axis=0)
        elif aggregation == 'mean':
            line = np.mean(performance, axis=0)
            std = np.std(performance, axis=0)*scale_std
            upper = line + std
            lower = line - std
        else:
            raise ValueError()

        # Plot mean and std
        if scale_std >= 0:
            ax1.fill_between(time_list[idx], lower, upper,
                             facecolor=color, alpha=0.3, edgecolor=color)
        ax1.plot(time_list[idx], line, color=color,
                 linewidth=int(properties["linewidth"]), linestyle=linestyle,
                 marker=marker, markersize=int(properties["markersize"]),
                 label=name_list[idx], markevery=0.1
                 )
        print(time_list[idx], line)

        # Get limits
        # For y_min we always take the lowest value
        auto_y_min = min(min(lower), auto_y_min)
        auto_y_max = max(max(upper), auto_y_max)

        auto_x_min = min(time_list[idx][0], auto_x_min)
        auto_x_max = max(time_list[idx][-1], auto_x_max)

    # Describe axes
    if logy:
        ax1.set_ylabel("log10(%s)" % ylabel, fontsize=properties["labelfontsize"])
    else:
        ax1.set_ylabel("%s" % ylabel, fontsize=properties["labelfontsize"])

    if logx:
        ax1.set_xlabel("time [sec]", fontsize=properties["labelfontsize"])
        ax1.set_xscale("log")
        auto_x_min = max(0.1, auto_x_min)
    else:
        ax1.set_xlabel("time [sec]")

    leg = ax1.legend(loc='best',  # loc=(0.01, 0.47),
                     fancybox=True, prop={'size': int(properties["legendsize"])})
    leg.get_frame().set_alpha(0.8)

    # Set axes limits
    if y_max is None and y_min is not None:
        ax1.set_ylim([y_min, auto_y_max + 0.01*abs(auto_y_max - y_min)])
    elif y_max is not None and y_min is None:
        ax1.set_ylim([auto_y_min - 0.01*abs(auto_y_max - y_min), y_max])
    elif y_max is not None and y_min is not None and y_max > y_min:
        ax1.set_ylim([y_min, y_max])
    else:
        ax1.set_ylim([auto_y_min - 0.01*abs(auto_y_max - auto_y_min),
                      auto_y_max + 0.01*abs(auto_y_max - auto_y_min)])

    if x_max is None and x_min is not None:
        ax1.set_xlim([x_min - 0.1*abs(x_min), auto_x_max + 0.1*abs(auto_x_max)])
    elif x_max is not None and x_min is None:
        ax1.set_xlim([auto_x_min - 0.1*abs(auto_x_min), x_max + 0.1*abs(x_max)])
    elif x_max is not None and x_min is not None and x_max > x_min:
        ax1.set_xlim([x_min, x_max])
    else:
        ax1.set_xlim([auto_x_min, auto_x_max + 0.1*abs(auto_x_min - auto_x_max)])

    # Save or show
    tight_layout()
    # subplots_adjust(top=0.85)
    if save != "":
        print("Save plot to %s" % save)
        savefig(save, dpi=int(properties['dpi']), facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1)
    else:
        show()


def main():
    prog = "python merge_performance_different_times.py <WhatIsThis> " \
           "one/or/many/*ClassicValidationResults*.csv"
    description = "Merge results to one csv"

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
                        default="", help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default=None, help="Optional supertitle for plot")
    parser.add_argument("--maxvalue", dest="maxvalue", type=float,
                        default=sys.maxsize, help="Replace all values higher than this?")
    parser.add_argument("--ylabel", dest="ylabel",
                        default="Minfunction value", help="y label")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        default=False, help="print number of runs on plot")

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
    file_list, name_list = plot_util.get_file_and_name_list(unknown, match_file='.csv')
    print("ptpfc file_ist", file_list)
    print("ptpfc name_ist", name_list)
    for idx in range(len(name_list)):
        # Jinu comment: file_list[idx] should contain only one file!
        assert len(file_list[idx]) == 1, "%s" % str(file_list[idx])
        print("%20s contains %d file(s)" % (name_list[idx], len(file_list[idx])))

    times = list()
    performances = list()
    for idx, name in enumerate(name_list):
        t = None
        p = None
        print("Processing %s" % name)
        fh = open(file_list[idx][0], 'r')
        reader = csv.reader(fh)
        for row in reader:
            # Delete this
            print("row: ", row)
            if t is None:
                # first row
                p = list([list() for i in range(len(row)-1)])
                t = list()
                continue
            t.append(float(row[0]))
            del row[0]
            [p[i].append(float(row[i])) for i in range(len(row))]
        times.append(t)
        performances.append(p)

    # Sort names alphabetical as done here:
    # http://stackoverflow.com/questions/15610724/sorting-multiple-lists-in-python-based-on-sorting-of-a-single-list
    sorted_lists = sorted(zip(name_list, times, performances), key=lambda x: x[0])
    name_list, times, performances = [[x[i] for x in sorted_lists] for i in range(3)]

    prop = {}
    args_dict = vars(args)
    for key in defaults:
        prop[key] = args_dict[key]

    plot_optimization_trace(time_list=times, performance_list=performances,
                            title=args.title, name_list=name_list, ylabel=args.ylabel,
                            logy=args.logy, logx=args.logx, save=args.save,
                            y_min=args.ymin, y_max=args.ymax, x_min=args.xmin,
                            x_max=args.xmax, properties=prop, scale_std=1,
                            scale_mean=100)


if __name__ == "__main__":
    main()
