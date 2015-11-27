from __future__ import print_function

import argparse
import cPickle
import itertools
import os
import StringIO
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
    import sklearn.metrics.pairwise
except:
    print("Failed to load TSNE, probably you're using sklearn 0.14.X")

from pyMetaLearn.metalearning.meta_base import MetaBase
import pyMetaLearn.metalearning.create_datasets
import pyMetaLearn.data_repositories.openml.apiconnector


def load_dataset(dataset, dataset_directory):
    dataset_dir = os.path.abspath(os.path.join(dataset_directory, dataset))
    fh = open(os.path.join(dataset_dir, dataset + ".pkl"))
    ds = cPickle.load(fh)
    fh.close()
    data_frame = ds.convert_arff_structure_to_pandas(ds
                                                     .get_unprocessed_files())
    class_ = data_frame.keys()[-1]
    attributes = data_frame.keys()[0:-1]
    X = data_frame[attributes]
    Y = data_frame[class_]
    return X, Y


def plot_metafeatures(metafeatures_plot_dir, metafeatures, metafeatures_times,
                      runs, method='pca', seed=1, depth=1, distance='l2'):
    """Project datasets in a 2d space and plot them.

    arguments:
      * metafeatures_plot_dir: a directory to save the generated plots
      * metafeatures: a pandas Dataframe from the MetaBase
      * runs: a dictionary of runs from the MetaBase
      * method: either pca or t-sne
      * seed: only used for t-sne
      * depth: if 1, a one-step look-ahead is performed
    """
    if type(metafeatures) != pd.DataFrame:
        raise ValueError("Argument metafeatures must be of type pd.Dataframe "
                         "but is %s" % str(type(metafeatures)))

    ############################################################################
    # Write out the datasets and their size as a TEX table
    # TODO put this in an own function
    dataset_tex = StringIO.StringIO()
    dataset_tex.write('\\begin{tabular}{lrrr}\n')
    dataset_tex.write('\\textbf{Dataset name} & '
                      '\\textbf{\#features} & '
                      '\\textbf{\#patterns} & '
                      '\\textbf{\#classes} \\\\\n')

    num_features = []
    num_instances = []
    num_classes = []

    for dataset in sorted(metafeatures.index):
        dataset_tex.write('%s & %d & %d & %d \\\\\n' % (
                        dataset.replace('larochelle_etal_2007_', '').replace(
                            '_', '-'),
                        metafeatures.loc[dataset]['number_of_features'],
                        metafeatures.loc[dataset]['number_of_instances'],
                        metafeatures.loc[dataset]['number_of_classes']))
        num_features.append(metafeatures.loc[dataset]['number_of_features'])
        num_instances.append(metafeatures.loc[dataset]['number_of_instances'])
        num_classes.append(metafeatures.loc[dataset]['number_of_classes'])

    dataset_tex.write('Minimum & %.1f & %.1f & %.1f \\\\\n' %
        (np.min(num_features), np.min(num_instances), np.min(num_classes)))
    dataset_tex.write('Maximum & %.1f & %.1f & %.1f \\\\\n' %
        (np.max(num_features), np.max(num_instances), np.max(num_classes)))
    dataset_tex.write('Mean & %.1f & %.1f & %.1f \\\\\n' %
        (np.mean(num_features), np.mean(num_instances), np.mean(num_classes)))

    dataset_tex.write('10\\%% quantile & %.1f & %.1f & %.1f \\\\\n' % (
        np.percentile(num_features, 10), np.percentile(num_instances, 10),
        np.percentile(num_classes, 10)))
    dataset_tex.write('90\\%% quantile & %.1f & %.1f & %.1f \\\\\n' % (
        np.percentile(num_features, 90), np.percentile(num_instances, 90),
        np.percentile(num_classes, 90)))
    dataset_tex.write('median & %.1f & %.1f & %.1f \\\\\n' % (
        np.percentile(num_features, 50), np.percentile(num_instances, 50),
        np.percentile(num_classes, 50)))
    dataset_tex.write('\\end{tabular}')
    dataset_tex.seek(0)

    dataset_tex_output = os.path.join(metafeatures_plot_dir, 'datasets.tex')
    with open(dataset_tex_output, 'w') as fh:
        fh.write(dataset_tex.getvalue())

    ############################################################################
    # Write out a list of metafeatures, each with the min/max/mean
    # calculation time and the min/max/mean value
    metafeatures_tex = StringIO.StringIO()
    metafeatures_tex.write('\\begin{tabular}{lrrrrrr}\n')
    metafeatures_tex.write('\\textbf{Metafeature} & '
                      '\\textbf{Minimum} & '
                      '\\textbf{Mean} & '
                      '\\textbf{Maximum} &'
                      '\\textbf{Minimum time} &'
                      '\\textbf{Mean time} &'
                      '\\textbf{Maximum time} '
                      '\\\\\n')

    for mf_name in sorted(metafeatures.columns):
        metafeatures_tex.write('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\\n'
                               % (mf_name.replace('_', '-'),
                                  metafeatures.loc[:,mf_name].min(),
                                  metafeatures.loc[:,mf_name].mean(),
                                  metafeatures.loc[:,mf_name].max(),
                                  metafeature_times.loc[:, mf_name].min(),
                                  metafeature_times.loc[:, mf_name].mean(),
                                  metafeature_times.loc[:, mf_name].max()))

    metafeatures_tex.write('\\end{tabular}')
    metafeatures_tex.seek(0)

    metafeatures_tex_output = os.path.join(metafeatures_plot_dir, 'metafeatures.tex')
    with open(metafeatures_tex_output, 'w') as fh:
        fh.write(metafeatures_tex.getvalue())

    # Without this scaling the transformation for visualization purposes is
    # useless
    metafeatures = metafeatures.copy()
    X_min = np.nanmin(metafeatures, axis=0)
    X_max = np.nanmax(metafeatures, axis=0)
    metafeatures = (metafeatures - X_min) / (X_max - X_min)

    # PCA
    if method == 'pca':
        pca = PCA(2)
        transformation = pca.fit_transform(metafeatures.values)

    elif method == 't-sne':
        if distance == 'l2':
            distance_matrix = sklearn.metrics.pairwise.pairwise_distances(
                metafeatures.values, metric='l2')
        elif distance == 'l1':
            distance_matrix = sklearn.metrics.pairwise.pairwise_distances(
                metafeatures.values, metric='l1')
        elif distance == 'runs':
            names_to_indices = dict()
            for metafeature in metafeatures.index:
                idx = len(names_to_indices)
                names_to_indices[metafeature] = idx

            X, Y = pyMetaLearn.metalearning.create_datasets\
                .create_predict_spearman_rank(metafeatures, runs,
                                              'combination')
            # Make a metric matrix out of Y
            distance_matrix = np.zeros((metafeatures.shape[0],
                                        metafeatures.shape[0]), dtype=np.float64)

            for idx in Y.index:
                dataset_names = idx.split("_")
                d1 = names_to_indices[dataset_names[0]]
                d2 = names_to_indices[dataset_names[1]]
                distance_matrix[d1][d2] = Y.loc[idx]
                distance_matrix[d2][d1] = Y.loc[idx]

        else:
            raise NotImplementedError()

        # For whatever reason, tsne doesn't accept l1 metric
        tsne = TSNE(random_state=seed, perplexity=50, verbose=1)
        transformation = tsne.fit_transform(distance_matrix)

    # Transform the transformation back to range [0, 1] to ease plotting
    transformation_min = np.nanmin(transformation, axis=0)
    transformation_max = np.nanmax(transformation, axis=0)
    transformation = (transformation - transformation_min) / \
                     (transformation_max - transformation_min)
    print(transformation_min, transformation_max)

    #for i, dataset in enumerate(directory_content):
    #    print dataset, meta_feature_array[i]
    fig = plt.figure(dpi=600, figsize=(12, 12))
    ax = plt.subplot(111)

    # The dataset names must be aligned at the borders of the plot in a way
    # the arrows don't cross each other. First, define the different slots
    # where the labels will be positioned and then figure out the optimal
    # order of the labels
    slots = []
    # 25 datasets on the top y-axis
    slots.extend([(-0.1 + 0.05 * i, 1.1) for i in range(25)])
    # 24 datasets on the right x-axis
    slots.extend([(1.1, 1.05 - 0.05 * i) for i in range(24)])
    # 25 datasets on the bottom y-axis
    slots.extend([(-0.1 + 0.05 * i, -0.1) for i in range(25)])
    # 24 datasets on the left x-axis
    slots.extend([(-0.1, 1.05 - 0.05 * i) for i in range(24)])

    # Align the labels on the outer axis
    labels_top = []
    labels_left = []
    labels_right = []
    labels_bottom = []

    for values in zip(metafeatures.index,
                      transformation[:, 0], transformation[:, 1]):
        label, x, y = values
        # Although all plot area goes up to 1.1, 1.1, the range of all the
        # points lies inside [0,1]
        if x >= y and x < 1.0 - y:
            labels_bottom.append((x, label))
        elif x >= y and x >= 1.0 - y:
            labels_right.append((y, label))
        elif y > x and x <= 1.0 -y:
             labels_left.append((y, label))
        else:
            labels_top.append((x, label))

    # Sort the labels according to their alignment
    labels_bottom.sort()
    labels_left.sort()
    labels_left.reverse()
    labels_right.sort()
    labels_right.reverse()
    labels_top.sort()

    # Build an index label -> x, y
    points = {}
    for values in zip(metafeatures.index,
                      transformation[:, 0], transformation[:, 1]):
        label, x, y = values
        points[label] = (x, y)

    # Find out the final positions...
    positions_top = {}
    positions_left = {}
    positions_right = {}
    positions_bottom = {}

    # Find the actual positions
    for i, values in enumerate(labels_bottom):
        y, label = values
        margin = 1.2 / len(labels_bottom)
        positions_bottom[label] = (-0.05 + i * margin, -0.1,)
    for i, values in enumerate(labels_left):
        x, label = values
        margin = 1.2 / len(labels_left)
        positions_left[label] = (-0.1, 1.1 - i * margin)
    for i, values in enumerate(labels_top):
        y, label = values
        margin = 1.2 / len(labels_top)
        positions_top[label] = (-0.05 + i * margin, 1.1)
    for i, values in enumerate(labels_right):
        y, label = values
        margin = 1.2 / len(labels_right)
        positions_right[label] = (1.1, 1.05 - i * margin)

    # Do greedy resorting if it decreases the number of intersections...
    def resort(label_positions, marker_positions, maxdepth=1):
        # TODO: are the inputs dicts or lists
        # TODO: two-step look-ahead
        def intersect(start1, end1, start2, end2):
            # Compute if there is an intersection, for the algorithm see
            # Computer Graphics by F.S.Hill

            # If one vector is just a point, it cannot intersect with a line...
            for v in [start1, start2, end1, end2]:
                if not np.isfinite(v).all():
                    return False     # Obviously there is no intersection

            def perpendicular(d):
                return np.array((-d[1], d[0]))

            d1 = end1 - start1      # denoted b
            d2 = end2 - start2      # denoted d
            d2_1 = start2 - start1  # denoted c
            d1_perp = perpendicular(d1)   # denoted by b_perp
            d2_perp = perpendicular(d2)   # denoted by d_perp

            t = np.dot(d2_1, d2_perp) / np.dot(d1, d2_perp)
            u = - np.dot(d2_1, d1_perp) / np.dot(d2, d1_perp)

            if 0 <= t <= 1 and 0 <= u <= 1:
                return True    # There is an intersection
            else:
                return False     # There is no intersection

        def number_of_intersections(label_positions, marker_positions):
            num = 0
            for key1, key2 in itertools.permutations(label_positions, r=2):
                s1 = np.array(label_positions[key1])
                e1 = np.array(marker_positions[key1])
                s2 = np.array(label_positions[key2])
                e2 = np.array(marker_positions[key2])
                if intersect(s1, e1, s2, e2):
                    num += 1
            return num

        # test if swapping two lines would decrease the number of intersections
        # TODO: if this was done with a datastructure different than dicts,
        # it could be much faster, because there is a lot of redundant
        # computing performed in the second iteration
        def swap(label_positions, marker_positions, depth=0,
                 maxdepth=maxdepth, best_found=sys.maxint):
            if len(label_positions) <= 1:
                return

            two_step_look_ahead = False
            while True:
                improvement = False
                for key1, key2 in itertools.combinations(label_positions, r=2):
                    before = number_of_intersections(label_positions, marker_positions)
                    # swap:
                    tmp = label_positions[key1]
                    label_positions[key1] = label_positions[key2]
                    label_positions[key2] = tmp
                    if depth < maxdepth and two_step_look_ahead:
                        swap(label_positions, marker_positions,
                             depth=depth+1, best_found=before)

                    after = number_of_intersections(label_positions, marker_positions)

                    if best_found > after and before > after:
                        improvement = True
                        print(before, after)
                        print("Depth %d: Swapped %s with %s" %
                              (depth, key1, key2))
                    else:       # swap back...
                        tmp = label_positions[key1]
                        label_positions[key1] = label_positions[key2]
                        label_positions[key2] = tmp

                    if after == 0:
                        break

                # If it is not yet sorted perfectly, do another pass with
                # two-step lookahead
                if before == 0:
                    print("Sorted perfectly...")
                    break
                print(depth, two_step_look_ahead)
                if two_step_look_ahead:
                    break
                if maxdepth == depth:
                    print("Reached maximum recursion depth...")
                    break
                if not improvement and depth < maxdepth:
                    print("Still %d errors, trying two-step lookahead" % before)
                    two_step_look_ahead = True

        swap(label_positions, marker_positions, maxdepth=maxdepth)

    resort(positions_bottom, points, maxdepth=depth)
    resort(positions_left, points, maxdepth=depth)
    resort(positions_right, points, maxdepth=depth)
    resort(positions_top, points, maxdepth=depth)

    # Helper function
    def plot(x, y, label_x, label_y, label, ha, va, relpos, rotation=0):
        ax.scatter(x, y, marker='o', label=label, s=80, linewidths=0.1,
                   color='blue', edgecolor='black')

        label = label.replace('larochelle_etal_2007_', '')

        x = ax.annotate(label, xy=(x, y), xytext=(label_x, label_y),
                    ha=ha, va=va, rotation=rotation,
                    bbox=dict(boxstyle='round', fc='gray', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', color='black',
                                    relpos=relpos))

    # Do the plotting
    for i, key in enumerate(positions_bottom):
        x, y = positions_bottom[key]
        plot(points[key][0], points[key][1], x, y,
             key, ha='right', va='top', rotation=45, relpos=(1, 1))
    for i, key in enumerate(positions_left):
        x, y = positions_left[key]
        plot(points[key][0], points[key][1], x, y, key,
             ha='right', va='top', rotation=45, relpos=(1, 1))
    for i, key in enumerate(positions_top):
        x, y = positions_top[key]
        plot(points[key][0], points[key][1], x, y, key,
             ha='left', va='bottom', rotation=45, relpos=(0, 0))
    for i, key in enumerate(positions_right):
        x, y = positions_right[key]
        plot(points[key][0], points[key][1], x, y, key,
             ha='left', va='bottom', rotation=45, relpos=(0, 0))

    # Resize everything
    box = ax.get_position()
    remove = 0.05 * box.width
    ax.set_position([box.x0 + remove, box.y0 + remove,
                     box.width - remove*2, box.height - remove*2])

    locs_x = ax.get_xticks()
    locs_y = ax.get_yticks()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    plt.savefig(os.path.join(metafeatures_plot_dir, "pca.png"))
    plt.savefig(os.path.join(metafeatures_plot_dir, "pca.pdf"))
    plt.clf()

    # Relation of features to each other...
    #correlations = []
    #for mf_1, mf_2 in itertools.combinations(metafeatures.columns, 2):

    #    x = metafeatures.loc[:, mf_1]
    #    y = metafeatures.loc[:, mf_2]
    #    rho, p = scipy.stats.spearmanr(x, y)
    #    correlations.append((rho, "%s-%s" % (mf_1, mf_2)))

        # plt.figure()
        # plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
        # plt.plot(x, y, "x")
        # plt.xlabel(mf_1)
        # plt.ylabel(mf_2)
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # plt.savefig(os.path.join(target_directory, mf_1 + "__" + mf_2 + "
        # .png"))
        # plt.close()

    #correlations.sort()
    #for cor in correlations:
        #print cor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=str)
    parser.add_argument("--runs", type=str)
    parser.add_argument("experiment_directory", type=str)
    parser.add_argument("-m", "--method", default='pca',
                        choices=['pca', 't-sne'],
                        help="Dimensionality reduction method")
    parser.add_argument("--distance", choices=[None, 'l1', 'l2', 'runs'],
                        default='l2')
    parser.add_argument("-s", "--seed", default=1, type=int)
    parser.add_argument("-d", "--depth", default=0, type=int)
    parser.add_argument("--subset", default='all', choices=['all', 'pfahringer_2000_experiment1'])
    args = parser.parse_args()

    with open(args.tasks) as fh:
        task_files_list = fh.readlines()
    # Load all the experiment run data only if needed
    if args.distance == 'runs':
        with open(args.runs) as fh:
            experiments_file_list = fh.readlines()
    else:
        experiments_file_list = StringIO.StringIO()
        for i in range(len(task_files_list)):
            experiments_file_list.write("\n")
        experiments_file_list.seek(0)

    pyMetaLearn.data_repositories.openml.apiconnector.set_local_directory(
        args.experiment_directory)
    meta_base = MetaBase(task_files_list, experiments_file_list)
    metafeatures = meta_base.get_all_metafeatures_as_pandas(
        metafeature_subset=args.subset)
    metafeature_times = meta_base.get_all_metafeatures_times_as_pandas(
        metafeature_subset=args.subset)

    #if args.subset:
    #    metafeatures = metafeatures.loc[:,subsets[args.subset]]
    #    metafeature_times = metafeature_times.loc[:,subsets[args.subset]]

    runs = meta_base.get_all_runs()

    general_plot_directory = os.path.join(args.experiment_directory, "plots")
    try:
        os.mkdir(general_plot_directory)
    except:
        pass
    metafeatures_plot_dir = os.path.join(general_plot_directory, "metafeatures")
    try:
        os.mkdir(metafeatures_plot_dir)
    except:
        pass

    plot_metafeatures(metafeatures_plot_dir, metafeatures, metafeature_times,
                      runs, method=args.method, seed=args.seed,
                      depth=args.depth, distance=args.distance)


