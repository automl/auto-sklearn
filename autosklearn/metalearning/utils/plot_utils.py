from collections import OrderedDict
import copy
import cPickle
import glob
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy.stats
import sys

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.statistics as statistics


def find_ground_truth(globstring):
    glob_results = glob.glob(globstring)
    if len(glob_results) > 1:
        raise Exception("There must be only one ground truth directory for "
                        "%s" % globstring)
    elif len(glob_results) == 0:
        print("Found no ground truth for %s" % globstring)
        return None

    with open(glob_results[0]) as fh:
        trials = cPickle.load(fh)

    return trials


def plot_rankings(trial_list, name_list, optimum=0, title="", log=False,
                  save="", y_min=0, y_max=0, cut=sys.maxint, figsize=(16, 6),
                  legend_ncols=4, bootstrap_samples=500):
    rankings = calculate_rankings(trial_list, name_list,
                                  bootstrap_samples, cut=cut)
    plot_ranking(rankings, optimum=optimum, title=title, log=log,
                 save=save, y_min=y_min, y_max=y_max, cut=cut,
                 figsize=figsize, legend_ncols=legend_ncols)


def calculate_rankings(trial_list, name_list, bootstrap_samples=500, cut=50):
    bootstrap_samples = int(bootstrap_samples)
    optimizers = [name[0] for name in name_list]
    pickles = plot_util.load_pickles(name_list, trial_list)
    rankings = dict()

    rs = np.random.RandomState(1)

    combinations = []
    for i in range(bootstrap_samples):
        combination = []
        target = len(optimizers)
        maximum = [len(pickles[name]) for name in optimizers]
        for idx in range(target):
            combination.append(rs.randint(maximum[idx]))
        combinations.append(np.array(combination))

    for optimizer in optimizers:
        rankings[optimizer] = np.zeros((cut+1,), dtype=np.float64)
        rankings[optimizer][0] = np.mean(range(1, len(optimizers) + 1))

    for i in range(1, cut+1):
        num_products = 0

        for combination in combinations:

            ranks = scipy.stats.rankdata(
                [np.round(
                    plot_util.get_best(pickles[optimizers[idx]][number], i), 5)
                 for idx, number in enumerate(combination)])
            num_products += 1
            for j, optimizer in enumerate(optimizers):
                rankings[optimizer][i] += ranks[j]

        for optimizer in optimizers:
            rankings[optimizer][i] = rankings[optimizer][i] / num_products

    return rankings


def plot_ranking(rankings, optimum=0, title="", log=False,
                 save="", y_min=0, y_max=0, figsize=(16, 6), legend_ncols=4,
                 colors=None, markers=None, markersize=6, linewidth=3):
    # check if all optimizers have the same number of runs
    # if np.mean([name[1] for name in name_list]) != name_list[0][1]:
    #    raise Exception("All optimizers must have the same numbers of "
    #                    "experiment runs! %s" % name_list)


    fig = plt.figure(dpi=600, figsize=figsize)
    ax = plt.subplot(111)

    if colors is None:
        colors = plot_util.get_plot_colors()
    if markers is None:
        markers = plot_util.get_empty_iterator()

    for i, optimizer in enumerate(rankings):
        ax.plot(range(0, rankings[optimizer].shape[0]),
                rankings[optimizer],
                marker=markers.next(), markersize=markersize,
                color=colors.next(), linewidth=linewidth,
                label=optimizer.replace("\\", "").
                replace("learned", "d_c").replace("l1", "d_p"))

    ax.legend(loc='upper center',  #bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=legend_ncols,
              labelspacing=0.25, fontsize=12)
    ax.set_xlabel("#Function evaluations")
    ax.set_ylabel("Average rank")
    box = ax.get_position()

    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])

    if save != "":
        plt.savefig(save, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    plt.close(fig)


def get_summed_wins_of_optimizers(trial_list_per_dataset,
                                  name_list_per_dataset,
                                  cut=sys.maxint):
    with open(trial_list_per_dataset[0][0][0]) as fh:
        probing_trial = cPickle.load(fh)
    cut = min(cut, len(probing_trial['trials']))
    # TODO remove this hack!
    cut = 50

    optimizers = []
    for name_list in name_list_per_dataset:
        optimizers.extend([name[0] for name in name_list])
    optimizers = set(optimizers)
    optimizers = list(optimizers)
    optimizers.sort()

    if cut == sys.maxint:
        raise ValueError("You must specify a cut value!")

    summed_wins_of_optimizer = \
        [np.zeros((len(optimizers), len(optimizers))) for i in range(cut + 1)]

    for pkl_list, name_list in itertools.izip(trial_list_per_dataset,
                                              name_list_per_dataset):
        # #######################################################################
        # Statistical stuff for one dataset
        for c in range(1, cut + 1):
            # TODO: this function call can be done in parallel
            wins_of_optimizer = statistics.get_pairwise_wins(pkl_list,
                                                             name_list, cut=c)

            # It can happen that not all optimizers are in all lists,
            # therefore we have to check this first
            for opt1_idx, key in enumerate(optimizers):
                for opt2_idx, key2 in enumerate(optimizers):
                    if key in wins_of_optimizer:
                        tmp_dict = wins_of_optimizer[key]
                        if key2 in tmp_dict:
                            summed_wins_of_optimizer[c][opt1_idx][opt2_idx] += \
                                wins_of_optimizer[key][key2]

    return optimizers, summed_wins_of_optimizer


def plot_summed_wins_of_optimizers(trial_list_per_dataset,
                                   name_list_per_dataset,
                                   save="",  cut=sys.maxint,
                                   figsize=(16, 4), legend_ncols=3,
                                   colors=None, linewidth=3,
                                   markers=None, markersize=6):
    # TODO colors should be a function handle which returns an Iterable!

    # This is a hack
    cut = 50
    optimizers, summed_wins_of_optimizer = get_summed_wins_of_optimizers(
        trial_list_per_dataset, name_list_per_dataset)

    # Make a copy of the colors iterator, because we need it more than once!
    if colors is not None:
        if not isinstance(colors, itertools.cycle):
            raise TypeError()
        else:
            color_values = list()
            for i in range(10):
                r, g, b = colors.next()
                color_values.append((r, g, b))

    if markers is not None:
        if not isinstance(markers, itertools.cycle):
            raise TypeError()
        else:
            marker_values = list()
            for i in range(10):
                marker_values.append(markers.next())

    ################################################################################
    # Plot statistics
    for opt1_idx, key in enumerate(optimizers):
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, dpi=600,
                                       figsize=figsize)

        if colors is None:
            colors_ = plot_util.get_plot_colors()
        else:
            colors_ = itertools.cycle(color_values)

        if markers is None:
            markers_ = plot_util.get_empty_iterator()
        else:
            markers_ = itertools.cycle(marker_values)

        y_max = 0.

        for opt2_idx, key2 in enumerate(optimizers):
            if opt1_idx == opt2_idx:
                continue

            y = []
            y1 = []
            for i in range(0, cut+1):
                y.append(summed_wins_of_optimizer[i][opt1_idx, opt2_idx]
                         / len(trial_list_per_dataset) * 100)
                y1.append(- summed_wins_of_optimizer[i][opt2_idx, opt1_idx]
                          / len(trial_list_per_dataset) * 100)

            y_max_tmp = max(np.max(y), np.max(np.abs(y1)))
            y_max_tmp = np.ceil(y_max_tmp * 10) / 10.
            y_max = max(y_max_tmp, y_max)

            label = "%s vs %s" % (key, key2)
            label = label.replace("learned", "d_c").replace("l1", "d_p")
            color = colors_.next()
            marker = markers_.next()
            ax0.plot(range(0, cut+1), y, color=color, label=label,
                     linewidth=linewidth, marker=marker, markersize=markersize)
            ax1.plot(range(0, cut+1), y1, color=color, label=label,
                     linewidth=linewidth, marker=marker, markersize=markersize)

        #handles, labels = ax1.get_legend_handles_labels()
        #fig.legend(handles, labels, loc="upper center", fancybox=True,
        #           ncol=legend_ncols, shadow=True)

        ax0.set_xlim((0, cut))
        ax0.set_ylim((0, y_max))
        ax0.set_ylabel("Significant wins (%)")
        ax1.set_xlim((0, cut))
        ax1.set_ylim((-y_max, 0))
        ax1.set_ylabel("Significant losses (%)")
        yticklabels = ax1.get_yticks().tolist()
        #print yticklabels, [item.get_text() for item in yticklabels]
        ax1.set_yticklabels([-int(item) for item in yticklabels])

        ax1.legend(loc="best", fancybox=True, ncol=legend_ncols, shadow=True)

        plt.tight_layout()
        #plt.subplots_adjust(top=0.85)
        plt.xlabel("#Function evaluations")
        if save != "":
            plt.savefig(save % key, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches="tight", pad_inches=0.1)
        else:
            plt.show()
        plt.close(fig)


# In case we want to display more than the 9 colors present in HPOlib
colors = matplotlib.colors.cnames.copy()
del colors['white']
del colors['whitesmoke']
del colors['ghostwhite']
del colors['aliceblue']
del colors['azure']
del colors['bisque']
del colors['cornsilk']
del colors['floralwhite']
del colors['honeydew']
del colors['ivory']
del colors['lavenderblush']
del colors['lightyellow']
del colors['mintcream']


# Set up a dictionary with keys to all hyperparameter optimization algorithms
def get_all_available_optimizers():
    optimizers = OrderedDict()

    # Helper variables
    better_subset_names = {"all": "All",
                           "pfahringer_2000_experiment1": "Pfahringer",
                           "yogotama_2014": "Yogatama",
                           "bardenet_2013_boost": "Bardenet(Boosting)",
                           "bardenet_2013_nn": "Bardenet(NN)",
                           "subset0": "ForwardFS",
                           "subset1": "EmbeddedFS"}
    num_samples = range(1, 100)
    all_distances = ["random", "l1", "l2", "learned"]
    all_subsets = ["all", "pfahringer_2000_experiment1", "yogotama_2014",
                   "bardenet_2013_boost", "bardenet_2013_nn", "subset0",
                   "subset1"]

    # Add SMBO methods
    optimizers["SMAC"] = "%s/smac_2_06_01-dev_*/smac_2_06_01-dev.pkl"
    optimizers[
        "TPE"] = "%s/hyperopt_august2013_mod_*/hyperopt_august2013_mod.pkl"
    optimizers[
        "Spearmint"] = "%s/spearmint_gitfork_mod_*/spearmint_gitfork_mod.pkl"

    # Random search
    optimizers[
        "random"] = "%s/random_hyperopt_august2013_mod*/random_hyperopt_august2013_mod.pkl"

    for samples in num_samples:
        for dist in all_distances:  # No L2 metric
            for subset in all_subsets:
                if dist == "random" and subset != "all":
                    continue

                optimizers["MI-SMAC(%d,%s,%s)" % (
                    samples, dist, better_subset_names[subset])] = \
                    "%s/bootstrapped" + \
                    "%d_%s_%ssmac_warmstart_*/*smac_warmstart.pkl" % \
                    (samples, dist, subset)

                optimizers["MI-Spearmint(%d,%s,%s)" % (
                    samples, dist, better_subset_names[subset])] = \
                    "%s/" + "bootstrapped%d_%s_%sspearmint_gitfork_mod_" \
                            "*/*spearmint_gitfork_mod.pkl" % (
                                samples, dist, subset)

    return optimizers
