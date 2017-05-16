import itertools
import logging
import numpy as np
import pandas as pd
import scipy.stats


def create_regression_dataset(metafeatures, experiments):
    X = []
    X_indices = []
    Y = []
    for dataset_name in experiments:
        experiment = experiments[dataset_name]
        mf = metafeatures.loc[dataset_name]
        for i, run in enumerate(experiment):
            x1 = pd.Series(data=[run.params[param] for param in run.params],
                           index=run.params.keys())
            x2 = mf
            X.append(x1.append(x2))
            X_indices.append('%s_%d' % (dataset_name, i))
            Y.append(run.result)
    X = pd.DataFrame(X, index=X_indices)
    Y = pd.DataFrame(Y, index=X_indices)
    logging.info("X.shape %s", X.shape)
    logging.info("Y.shape %s", Y.shape)
    return X, Y


def create_predict_spearman_rank(metafeatures, experiments, iterator):
    X = []
    Y = []
    Y_names = []
    # Calculate the pairwise ranks between datasets
    dataset_names = [name for name in metafeatures.index]
    cross_product = []

    if iterator == "combination":
        for cross in itertools.combinations_with_replacement(dataset_names, r=2):
            cross_product.append(cross)
    elif iterator == "permutation":
        for cross in itertools.permutations(dataset_names, r=2):
            cross_product.append(cross)
    else:
        raise NotImplementedError(iterator)
    logging.info("Create spearman rank dataset without CV data and %s",
                iterator)
    logging.info("Using %d datasets", len(dataset_names))
    logging.info("This will results in %d training points", len(cross_product))

    # Create inputs and targets
    for cross in cross_product:
        name = "%s_%s" % (cross[0], cross[1])

        mf_1 = metafeatures.loc[cross[0]]
        mf_2 = metafeatures.loc[cross[1]]
        assert mf_1.dtype == np.float64
        assert mf_2.dtype == np.float64
        x = np.hstack((mf_1, mf_2))
        columns = metafeatures.columns.values
        index = np.hstack(("0_" + columns, "1_" + columns))
        x = pd.Series(data=x, name=name, index=index)
        X.append(x)

        experiments_1 = experiments[cross[0]]
        experiments_2 = experiments[cross[1]]
        assert len(experiments_1) == len(experiments_2), name

        responses_1 = np.zeros((len(experiments_1)), dtype=np.float64)
        responses_2 = np.zeros((len(experiments_1)), dtype=np.float64)

        for idx, zipped in enumerate(zip(
                sorted(experiments_1, key=lambda t: str(t.configuration)),
                sorted(experiments_2, key=lambda t: str(t.configuration)))):
            # Test if the order of the params is the same
            exp_1, exp_2 = zipped
            print(exp_1.configuration, exp_2.configuration)
            assert exp_1.configuration == exp_2.configuration,\
                (experiments_1, experiments_2)
            responses_1[idx] = exp_1.result if np.isfinite(exp_1.result) else 1
            responses_2[idx] = exp_2.result if np.isfinite(exp_2.result) else 1

        rho, p = scipy.stats.spearmanr(responses_1, responses_2)
        #rho, p = scipy.stats.kendalltau(responses_1, responses_2)
        if not np.isfinite(rho):
            rho = 0

        Y.append(rho)
        Y_names.append(name)

    X = pd.DataFrame(X)
    Y = pd.Series(Y, index=Y_names)
    logging.info("Metafeatures %s", metafeatures.shape)
    logging.info("X.shape %s", X.shape)
    logging.info("Y.shape %s", Y.shape)
    assert X.shape == (len(cross_product), metafeatures.shape[1] * 2), \
        (X.shape, (len(cross), metafeatures.shape[1] * 2))
    assert Y.shape == (len(cross_product), )
    # train sklearn regressor (tree) with 10fold CV
    indices = range(len(X))
    np_rs = np.random.RandomState(42)
    np_rs.shuffle(indices)
    X = X.iloc[indices]
    Y = Y.iloc[indices]
    return X, Y


def create_predict_spearman_rank_with_cv(cv_metafeatures, cv_experiments,
                                         iterator):
    X = []
    Y = []
    Y_names = []
    # Calculate the pairwise ranks between datasets
    dataset_names = [name for name in cv_metafeatures]
    cross_product = []
    folds_product = []

    if iterator == "combination":
        for cross in itertools.combinations_with_replacement(dataset_names, r=2):
            cross_product.append(cross)
        for folds in itertools.combinations_with_replacement(range(10), r=2):
            folds_product.append(folds)
    elif iterator == "permutation":
        for cross in itertools.permutations(dataset_names, r=2):
            cross_product.append(cross)
        for folds in itertools.permutations(range(10), r=2):
            folds_product.append(folds)
    else:
        raise NotImplementedError()

    logging.info("Create spearman rank dataset with CV data %s", iterator)
    logging.info("Using %d datasets", len(dataset_names))
    logging.info("This will results in %d training points",
        (len(cross_product) * len(folds_product)))
    logging.info("Length of dataset crossproduct %s", len(cross_product))
    logging.info("Length of folds crossproduct %s", len(folds_product))

    # Create inputs and targets
    for i, cross in enumerate(cross_product):
        print("%d/%d: %s" % (i, len(cross_product), cross),)
        for folds in folds_product:
            name = "%s-%d_%s-%d" % (cross[0], folds[0], cross[1], folds[1])
            mf_1 = cv_metafeatures[cross[0]][folds[0]]
            mf_2 = cv_metafeatures[cross[1]][folds[1]]
            assert mf_1.dtype == np.float64
            assert mf_2.dtype == np.float64
            x = np.hstack((mf_1, mf_2))
            columns = cv_metafeatures[cross[0]][folds[0]].index.values
            index = np.hstack(("0_" + columns, "1_" + columns))
            x = pd.Series(data=x, name=name, index=index)
            X.append(x)

            experiments_1 = cv_experiments[cross[0]][folds[0]]
            experiments_2 = cv_experiments[cross[1]][folds[1]]
            assert len(experiments_1) == len(experiments_2)

            responses_1 = np.zeros((len(experiments_1)), dtype=np.float64)
            responses_2 = np.zeros((len(experiments_1)), dtype=np.float64)
            for idx, zipped in enumerate(zip(experiments_1, experiments_2)):
                # Test if the order of the params is the same
                exp_1, exp_2 = zipped
                assert exp_1.params == exp_2.params
                responses_1[idx] = exp_1.result
                responses_2[idx] = exp_2.result

            rho, p = scipy.stats.spearmanr(responses_1, responses_2)
            # A nan is produced if all values of one of the response lists
            # are equal. This results in a division by zero. Because there is
            #  no correlation if all values are the same, rho is replaced by
            # zero...
            # It would probably be better to assign random ranks for equal
            # values, but scipy doesn't support this...
            if not np.isfinite(rho):
                rho = 0
            Y.append(rho)
            Y_names.append(name)

    X = pd.DataFrame(X)
    Y = pd.Series(Y, index=Y_names)
    logging.info("CV_Metafeatures %s", cv_metafeatures.shape)
    logging.info("X.shape %s", X.shape)
    logging.info("Y.shape %s", Y.shape)
    # train sklearn regressor (tree) with 10fold CV
    indices = range(len(X))
    np_rs = np.random.RandomState(42)
    np_rs.shuffle(indices)
    X = X.iloc[indices]
    Y = Y.iloc[indices]
    return X, Y

"""
def create_smac_warmstart_files(context, dataset, output_dir, num_warmstarts):
    runs_and_results = StringIO.StringIO()
    runs_and_results.write("Run Number,Run History Configuration ID,Instance ID,"
                           "Response Value (y),Censored?,Cutoff Time Used,Seed,"
                           "Runtime,Run Length,Run Result Code,Run Quality,SMAC"
                           " Iteration,SMAC Cumulative Runtime,Run Result,"
                           "Additional Algorithm Run Data,Wall Clock Time,\n")
    paramstrings = StringIO.StringIO()

    best_hyperparameters, distances = metalearner.metalearn_base(context)
    hp_list, name_list, dist_list = metalearner.assemble_best_hyperparameters_list(
        best_hyperparameters, distances)

    for i in range(len(hp_list)):
        print hp_list[i], name_list[i], dist_list[i]




def create_smac_files_file(cv_metafeatures, cv_experiments, dataset,
                           output_dir):
    runs_and_results = StringIO.StringIO()
    runs_and_results.write("Run Number,Run History Configuration ID,Instance ID,"
                           "Response Value (y),Censored?,Cutoff Time Used,Seed,"
                           "Runtime,Run Length,Run Result Code,Run Quality,SMAC"
                           " Iteration,SMAC Cumulative Runtime,Run Result,"
                           "Additional Algorithm Run Data,Wall Clock Time,\n")
    paramstrings = StringIO.StringIO()
    train_instances_file = StringIO.StringIO()
    feature_file = StringIO.StringIO()
    scenario_file = StringIO.StringIO()

    run_number = 1
    instance_number = 1


    # TODO: is it possible to get_value the openml dataset id?
    for dataset_number, name in enumerate(cv_experiments):

        for fold in cv_experiments[name]:
            configuration_id = 1
            iteration = int(run_number/2)
            # if name == dataset, we don't want to put the rundata in there
            # because we want to optimize for name
            if name != dataset:
                for exp in cv_experiments[name][fold]:
                    str = "%s,%s,%s,%f,0,108000,-1,%f,1,1,%f,%d,%f,SAT,Aditional data,%f" \
                        % (run_number, configuration_id, instance_number, exp.result, 1.0,
                           exp.result, iteration, float(run_number), 1.0)
                    runs_and_results.write(str + "\n")
                    run_number += 1
                    configuration_id += 1

            train_instances_file.write("%d-%d\n" % (dataset_number, fold))
            instance_number += 1
        if instance_number > 100:
            break

    configuration_id = 1
    for exp in cv_experiments[name][0]:
        paramstring = ", ".join(["%s='%s'" % (re.sub("^-", "",param),
            exp.params[param]) for param in exp.params])
        paramstrings.write("%d: %s\n" % (configuration_id, paramstring))

    with open(os.path.join(output_dir, "runs_and_results-it%d.csv" %
            iteration), "w") as fh:
        runs_and_results.seek(0)
        for line in runs_and_results:
            fh.write(line)
    with open(os.path.join(output_dir, "paramstrings-it%d.txt" % iteration),
              "w") as fh:
        paramstrings.seek(0)
        for line in paramstrings:
            fh.write(line)
    with open(os.path.join(output_dir, "instances-train.txt"),
              "w") as fh:
        train_instances_file.seek(0)
        for line in train_instances_file:
            fh.write(line)
"""


if __name__ == "__main__":
    pass
    """
    # TODO: right now, this is only done for one split, namely the split of
    # the directory we're inside...
    # TODO: this only works in a directory, in which a metaexperiment was
    # already run...
    parser = ArgumentParser()
    parser.add_argument("target_directory", type=str)
    args = parser.parse_args()

    target_directory = args.target_directory
    if not os.path.exists(target_directory):
        raise ValueError("Target directory %s does not exist." % target_directory)

    # Important, change into some directory in which an experiment was already
    # performed...
    context = metalearner.setup(None)
    metafeatures = context["metafeatures"]
    #cv_metafeatures = context["cv_metafeatures"]
    meta_base = context["meta_base"]
    #cv_meta_base = context["cv_meta_base"]

    savefile_prefix = "testfold_%d-%d" % (context["test_fold"],
                                          context["test_folds"])

    # Use only the pfahringer subset of the available metafeatures
    #columns = list()
    #columns.extend(mf.subsets["pfahringer_2000_experiment1"])
    #print columns
    #metafeatures = metafeatures.loc[:,columns]
    #for key in cv_metafeatures:
    #    cv_metafeatures[key] = cv_metafeatures[key].loc[columns,:]
    # savefile_prefix += "_pfahringer"

    # Remove class_probability_max from the set of metafeatures
    # columns = list()
    # metafeature_list = mf.subsets["all"]
    # metafeature_list.remove("class_probability_max")
    # metafeature_list.remove("class_probability_min")
    # metafeature_list.remove("class_probability_mean")
    # metafeature_list.remove("class_probability_std")
    # columns.extend(metafeature_list)
    # metafeatures = metafeatures.loc[:,columns]
    # for key in cv_metafeatures:
    #     cv_metafeatures[key] = cv_metafeatures[key].loc[columns,:]
    # savefile_prefix += "_woclassprobs"

    # Experiment is an OrderedDict, which has dataset names as keys
    # The values are lists of experiments(OrderedDict of params, response)
    experiments = meta_base.experiments
    #cv_experiments = cv_meta_base.experiments
    """
    """
    # Build the warmstart directory for SMAC, can be called with
    # ./smac --scenario-file <file> --seed 0 --warmstart <foldername>
    # needs paramstrings.txt and runs_and_results.txt
    # plain
    smac_bootstrap_output = "smac_bootstrap_plain"
    for dataset in cv_metafeatures:
        bootstraps = (2, 5, 10)
        distance = ("l1", "l2", "learned_distance")
        metafeature_subset = mf.subsets

        for num_bootstrap, dist, subset in itertools.product(
                bootstraps, distance, metafeature_subset, repeat=1):

            context["distance_measure"] = dist
            # TODO: somehow only get_value a metafeature subset

            dataset_output_dir = os.path.join(target_directory,
                smac_bootstrap_output, dataset +
                "_bootstrapped%d_%s_%s" % (num_bootstrap, dist, subset))
            if not os.path.exists(dataset_output_dir):
                os.mkdirs(dataset_output_dir)
            create_smac_warmstart_files(context, dataset, dataset_output_dir,
                                        num_warmstarts=num_bootstrap)
            break
    # with the adjustment of Yogotama and Mann

    """
    #X, Y = create_regression_dataset(metafeatures, experiments)
    #with open("regression_dataset.pkl", "w") as fh:
    #    cPickle.dump((X, Y, metafeatures), fh, -1)

    """
    # Calculate the metafeatures without the 10fold CV
    X, Y = create_predict_spearman_rank(metafeatures, experiments,
                                        iterator="permutation")
    spearman_rank_file = os.path.join(target_directory,
        savefile_prefix + "_spearman_rank_perm.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)

    X, Y = create_predict_spearman_rank(metafeatures, experiments,
                                        iterator="combination")
    spearman_rank_file = os.path.join(target_directory,
        savefile_prefix + "_spearman_rank_comb.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)

    print

    # Calculate the metafeatures for the 10fold CV...
    X, Y = create_predict_spearman_rank_with_cv(cv_metafeatures,
                                                cv_experiments,
                                                iterator="combination")
    spearman_rank_file = os.path.join(target_directory,
        savefile_prefix + "_spearman_rank_cv_comb.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)

    X, Y = create_predict_spearman_rank_with_cv(cv_metafeatures,
                                                cv_experiments,
                                                iterator="permutation")
    spearman_rank_file = os.path.join(target_directory,
        savefile_prefix + "_spearman_rank_cv_perm.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)
    """



