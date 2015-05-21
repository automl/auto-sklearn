from itertools import chain
import os
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from autosklearn.data import data_manager as data_manager
from autosklearn.metalearning import metalearning
from autosklearn.models import paramsklearn
from autosklearn.data import split_data
from autosklearn import submit_process
from autosklearn.util import stopwatch

from HPOlibConfigSpace.converters import pcs_parser
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

import autosklearn.util.logging_


def start_automl_on_dataset(basename, input_dir, tmp_dataset_dir, output_dir,
                            time_left_for_this_task, per_run_time_limit,
                            queue, log_dir=None,
                            initial_configurations_via_metalearning=25,
                            ensemble_size=1, ensemble_nbest=1):
    logger = autosklearn.util.logging_.get_logger(
        outputdir=log_dir, name="automl_%s" % basename)
    stop = stopwatch.StopWatch()
    stop.start_task(basename)
    stop.start_task("LoadData")

    # == Creating a data object with data and information about it
    logger.debug("======== Reading and converting data ==========")
    # Encoding the labels will be done after the metafeature calculation!
    loaded_data_manager = data_manager.DataManager(basename, input_dir,
                                                   verbose=True,
                                                   encode_labels=False)
    loaded_data_manager_str = str(loaded_data_manager).split("\n")
    for part in loaded_data_manager_str:
        logger.debug(part)

    # == Split dataset and store Data for the ensemble script
    X_train, X_ensemble, Y_train, Y_ensemble = split_data.split_data(
        loaded_data_manager.data['X_train'], loaded_data_manager.data['Y_train'])
    np.save(os.path.join(tmp_dataset_dir, "true_labels_ensemble.npy"), Y_ensemble)
    del X_train, X_ensemble, Y_train, Y_ensemble

    time_needed_to_load_data = stop.wall_elapsed(basename)
    time_left_after_reading = max(0, time_left_for_this_task -
                                  time_needed_to_load_data)
    logger.info("Remaining time after reading %s %5.2f sec" % (basename, time_left_after_reading))

    stop.stop_task("LoadData")
    # = Create a searchspace
    stop.start_task("CreateSearchSpace")
    searchspace_path = os.path.join(tmp_dataset_dir, "space.pcs")
    config_space = paramsklearn.get_configuration_space(loaded_data_manager.info)

    # Remove configurations we think are unhelpful
    classifiers = [# Never the best
                   'passive_aggressive',
                   'adaboost',
                   'gaussian_nb',
                   'k_nearest_neighbors',
                   'multinomial_nb',
                   # Only the best with one preprocessing method
                   'bernoulli_nb',
                   # Causes lots of crashes
                   'qda',
                   'lda']

    preprocessors = [  # The very slowest method
                       'kernel_pca']

    combinations = [  # Not too good
                      ("proj_logit", "fast_ica"),
                      ("proj_logit", "extra_trees_preproc_for_classification"),
                      ("proj_logit", "pca"),
                      ("proj_logit", "no_preprocessing"),
                      ("proj_logit", "liblinear_svc_preprocessor"),
                      ("proj_logit", "select_rates"),
                      ("proj_logit", "nystroem_sampler"),
                      ("proj_logit", "kitchen_sinks"),
                      ("sgd", "gem"),
                      ("sgd", "pca"),
                      ("sgd", "extra_trees_preproc_for_classification"),
                      ("decision_tree", "fast_ica"),
                      ("decision_tree", "gem"),
                      ("decision_tree", "pca"),
                      ("decision_tree", "liblinear_svc_preprocessor"),
                      ("decision_tree", "extra_trees_preproc_for_classification"),
                      ("libsvm_svc", "pca"),
                      ("libsvm_svc", "no_preprocessing"),
                      ("libsvm_svc", "select_percentile_classification"),
                      ("liblinear_svc", "gem"),
                      ("liblinear_svc", "pca"),
                      ("liblinear_svc", "fast_ica"),
                      ("liblinear_svc", "extra_trees_preproc_for_classification"),
                      ("gradient_boosting", "pca"),
                      ("gradient_boosting", "no_preprocessing"),
                      ("ridge", "pca"),
                      ("ridge", "fast_ica"),
                      ("ridge", "liblinear_svc_preprocessor"),
                      ("ridge", "extra_trees_preproc_for_classification"),
                      ("ridge", "select_percentile_classification"),
                      ("ridge", "select_rates"),
                      ("random_forest", "fast_ica"),
                      ("random_forest", "pca"),
                      ("random_forest", "liblinear_svc_preprocessor"),
                      ("extra_trees", "liblinear_svc_preprocessor"),


                      # Slow
                      ('random_forest', 'fast_ica'),
                      ('libsvm_svc', 'fast_ica'),
                      ('gradient_boosting', 'fast_ica'),
                      ('extra_trees', 'fast_ica'),
                      ('sgd', 'nystroem_sampler')]

    if loaded_data_manager.info['is_sparse'] == 1:
        classifiers.extend(['proj_logit', 'decision_tree',
                            'gradient_boosting'])

        preprocessors.append(['kitchen_sinks'])

        combinations.append([
            # Not really good
            ('libsvm_svc', 'truncatedSVD'),
            ('ridge', 'no_preprocessing'),
            ('ridge', 'select_percentile_classification'),
            ('ridge', 'select_rates'),
            ('libsvm_svc', 'liblinear_svc_preprocessor'),
            ('libsvm_svc', 'select_rates')])

    for classifier in classifiers:
        try:
            config_space.add_forbidden_clause(ForbiddenEqualsClause(
                config_space.get_hyperparameter("classifier"), classifier))
        except ValueError as e:
            logger.debug(e)
        except KeyError:
            pass

    for preprocessor in preprocessors:
        try:
            config_space.add_forbidden_clause(ForbiddenEqualsClause(
                config_space.get_hyperparameter("preprocessor"), preprocessor))
        except ValueError as e:
            logger.debug(e)

    for combination in combinations:
        try:
            config_space.add_forbidden_clause(ForbiddenAndConjunction(
                ForbiddenEqualsClause(config_space.get_hyperparameter("classifier"),
                                      combination[0]),
                ForbiddenEqualsClause(config_space.get_hyperparameter("preprocessor"),
                                      combination[1])
            ))
        except ValueError as e:
            logger.debug(e)
        except KeyError:
            pass

    sp_string = pcs_parser.write(config_space)
    fh = open(searchspace_path, 'w')
    fh.write(sp_string)
    fh.close()
    logger.debug("Searchspace written to %s" % searchspace_path)
    stop.stop_task("CreateSearchSpace")

    # == Calculate metafeatures
    stop.start_task("CalculateMetafeatures")
    categorical = [True if feat_type.lower() in ["categorical"] else False
                   for feat_type in loaded_data_manager.feat_type]

    if initial_configurations_via_metalearning <= 0:
        ml = None
    elif loaded_data_manager.info["task"].lower() in \
            ["multiclass.classification", "binary.classification"]:
        ml = metalearning.MetaLearning()
        logger.debug("Start calculating metafeatures for %s" %
                     loaded_data_manager.basename)
        ml.calculate_metafeatures_with_labels(loaded_data_manager.data["X_train"],
                                              loaded_data_manager.data["Y_train"],
                                              categorical=categorical,
                                              dataset_name=loaded_data_manager.basename)
    else:
        ml = None
        logger.critical("Metafeatures not calculated")
    stop.stop_task("CalculateMetafeatures")
    logger.debug("Calculating Metafeatures (categorical attributes) took %5.2f" % stop.wall_elapsed("CalculateMetafeatures"))

    stop.start_task("OneHot")
    loaded_data_manager.perform1HotEncoding()
    stop.stop_task("OneHot")

    if ml is None:
        initial_configurations = []
    elif loaded_data_manager.info["task"].lower() in \
            ["multiclass.classification", "binary.classification"]:
        stop.start_task("CalculateMetafeaturesEncoded")
        ml.calculate_metafeatures_encoded_labels(X_train=loaded_data_manager.data["X_train"],
                                                 Y_train=loaded_data_manager.data["Y_train"],
                                                 categorical=[False] * loaded_data_manager.data["X_train"].shape[0],
                                                 dataset_name=loaded_data_manager.basename)
        stop.stop_task("CalculateMetafeaturesEncoded")
        logger.debug(
            "Calculating Metafeatures (encoded attributes) took %5.2fsec" %
            stop.wall_elapsed("CalculateMetafeaturesEncoded"))

        logger.debug(ml._metafeatures_labels.__repr__(verbosity=2))
        logger.debug(ml._metafeatures_encoded_labels.__repr__(verbosity=2))

        # TODO check that Metafeatures only contain finite numbers!

        stop.start_task("InitialConfigurations")
        initial_configurations = ml.create_metalearning_string_for_smac_call(
            config_space, loaded_data_manager.basename, loaded_data_manager.info[
                'metric'], initial_configurations_via_metalearning)
        stop.stop_task("InitialConfigurations")
        logger.debug("Looking for initial configurations took %5.2fsec" %
                     stop.wall_elapsed("InitialConfigurations"))
        logger.info(
            "Time left for %s after finding initial configurations: %5.2fsec" %
            (basename, time_left_for_this_task - stop.wall_elapsed(basename)))
    else:
        initial_configurations = []
        logger.critical("Metafeatures encoded not calculated")

    # == Pickle the data manager
    stop.start_task("StoreDatamanager")
    data_manager_path = os.path.join(tmp_dataset_dir, basename + "_Manager.pkl")
    pickle.dump(loaded_data_manager, open(data_manager_path, 'w'), protocol=-1)
    logger.debug("Pickled Datamanager under %s" % data_manager_path)
    stop.stop_task("StoreDatamanager")

    # == RUN SMAC
    stop.start_task("runSmac")
    # = Create an empty instance file
    instance_file = os.path.join(tmp_dataset_dir, "instances.txt")
    fh = open(instance_file, 'w')
    fh.write("holdout")
    fh.close()
    logger.debug("Create instance file %s" % instance_file)

    # = Start SMAC
    dataset = os.path.join(input_dir, basename)
    time_left_for_smac = max(0, time_left_for_this_task - (stop.wall_elapsed(basename)))
    logger.debug("Start SMAC with %5.2fsec time left" % time_left_for_smac)
    proc_smac, smac_call = \
        submit_process.run_smac(dataset=dataset,
                                tmp_dir=tmp_dataset_dir,
                                searchspace=searchspace_path,
                                instance_file=instance_file,
                                limit=time_left_for_smac,
                                cutoff_time=per_run_time_limit,
                                initial_challengers=initial_configurations)
    logger.debug(smac_call)
    stop.stop_task("runSmac")

    # == RUN ensemble builder
    stop.start_task("runEnsemble")
    time_left_for_ensembles = max(0, time_left_for_this_task - (stop.wall_elapsed(basename)))
    logger.debug("Start Ensemble with %5.2fsec time left" % time_left_for_ensembles)
    proc_ensembles = \
        submit_process.run_ensemble_builder(tmp_dir=tmp_dataset_dir,
                                            dataset_name=basename,
                                            task_type=loaded_data_manager.info['task'],
                                            metric=loaded_data_manager.info['metric'],
                                            limit=time_left_for_ensembles,
                                            output_dir=output_dir,
                                            ensemble_size=ensemble_size)
    stop.stop_task("runEnsemble")

    queue.put([time_needed_to_load_data, data_manager_path,
               proc_smac, proc_ensembles])
    return
