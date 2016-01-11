import argparse
import ast
from collections import OrderedDict
import logging
import sys

import pandas as pd
import numpy as np
import sklearn.utils

from autosklearn.metalearning.metalearning.meta_base import MetaBase, Run
from autosklearn.metalearning.metalearning.kNearestDatasets.kND import KNearestDatasets

def test_function(params):
    return np.random.random()


class MetaLearningOptimizer(object):
    def __init__(self, dataset_name, configuration_space,
                 aslib_directory, distance='l1', seed=None, use_features='',
                 distance_kwargs=None, subset='all'):
        """Metalearning optimizer.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        configuration_space : ConfigSpace.configuration_space.ConfigurationSpace

        datasets_file : str
            Path to an aslib directory

        distance : str, "l1" or "l2" or "random"
            Distance function to be used by the kNearestDatasets algorithm.

        seed

        use_features

        metric_kwargs

        subset
        """
        self.dataset_name = dataset_name
        self.configuration_space = configuration_space
        self.aslib_dir = aslib_directory
        self.distance = distance
        self.seed = seed
        self.use_features = use_features
        self.distance_kwargs = distance_kwargs
        self.subset = subset
        self.kND = None     # For caching, makes things faster...

        self.meta_base = MetaBase(configuration_space, self.aslib_dir)
        self.logger = logging.getLogger(__name__)

    def perform_sequential_optimization(self, target_algorithm=test_function,
                                        time_budget=None,
                                        evaluation_budget=None):
        raise NotImplementedError("Right now this is not implemented due to "
                                  "timing issues.")
        time_taken = 0
        num_evaluations = 0
        history = []

        self.logger.info("Taking distance measure %s" % self.distance)
        while True:
            if time_budget is not None and time_taken >= time_budget:
                self.logger.info("Reached time budget. Exiting optimization.")
                break
            if evaluation_budget is not None and \
                    num_evaluations >= evaluation_budget:
                self.logger.info("Reached maximum number of evaluations. Exiting "
                            "optimization.")
                break

            params = self.metalearning_suggest(history)

            fixed_params = OrderedDict()
            # Hack to remove all trailing - from the params which are
            # accidently in the experiment pickle of the current HPOlib version
            for key in params:
                if key[0] == "-":
                    fixed_params[key[1:]] = params[key]
                else:
                    fixed_params[key] = params[key]

            self.logger.info("%d/%d, parameters: %s" % (num_evaluations,
                                                   evaluation_budget,
                                                   str(fixed_params)))
            result = target_algorithm(fixed_params)
            history.append(Run(params, result))
            num_evaluations += 1

        return min([run.result for run in history])

    def metalearning_suggest_all(self, exclude_double_configurations=True):
        """Return a list of the best hyperparameters of neighboring datasets"""
        # TODO check if _learn was called before!
        neighbors = self._learn(exclude_double_configurations)
        hp_list = []
        for neighbor in neighbors:
            try:
                configuration = \
                    self.meta_base.get_configuration_from_algorithm_index(
                    neighbor[2])
                self.logger.info("%s %s %s" % (neighbor[0], neighbor[1], configuration))
            except (KeyError):
                self.logger.warning("Configuration %s not found" % neighbor[2])
                continue

            hp_list.append(configuration)
        return hp_list

    def metalearning_suggest(self, history):
        """Suggest the next most promising hyperparameters which were not yet evaluated"""
        # TODO test the object in the history!
        neighbors = self._learn()
        # Iterate over all datasets which are sorted ascending by distance

        history_with_indices = []
        for run in history:
            history_with_indices.append(\
                self.meta_base.get_algorithm_index_from_configuration(run))

        for idx, neighbor in enumerate(neighbors):
            already_evaluated = False
            # Check if that dataset was already evaluated
            for run in history_with_indices:
                # If so, return to the outer loop

                if neighbor[2] == run:
                    already_evaluated = True
                    break

            if not already_evaluated:
                self.logger.info("Nearest dataset with hyperparameters of best value "
                            "not evaluated yet is %s with a distance of %f" %
                            (neighbor[0], neighbor[1]))
                return self.meta_base.get_configuration_from_algorithm_index(
                    neighbor[2])
        raise StopIteration("No more values available.")

    def _learn(self, exclude_double_configurations=True):
        dataset_metafeatures, all_other_metafeatures = self._get_metafeatures()

        # Remove metafeatures which could not be calculated for the target
        # dataset
        keep = []
        for idx in dataset_metafeatures.index:
            if np.isfinite(dataset_metafeatures.loc[idx]):
               keep.append(idx)

        dataset_metafeatures = dataset_metafeatures.loc[keep]
        all_other_metafeatures = all_other_metafeatures.loc[:,keep]

        # Do mean imputation of all other metafeatures
        all_other_metafeatures = all_other_metafeatures.fillna(
            all_other_metafeatures.mean())

        if self.kND is None:
            # In case that we learn our distance function, get_value the parameters for
            #  the random forest
            if self.distance_kwargs:
                rf_params = ast.literal_eval(self.distance_kwargs)
            else:
                rf_params = None

            # To keep the distance the same in every iteration, we create a new
            # random state
            random_state = sklearn.utils.check_random_state(self.seed)
            kND = KNearestDatasets(metric=self.distance,
                                   random_state=random_state,
                                   metric_params=rf_params)

            runs = dict()
            # TODO move this code to the metabase
            for task_id in all_other_metafeatures.index:
                try:
                    runs[task_id] = self.meta_base.get_runs(task_id)
                except KeyError:
                    # TODO should I really except this?
                    self.logger.warning("Could not find runs for instance %s" % task_id)
                    runs[task_id] = pd.Series([], name=task_id)
            runs = pd.DataFrame(runs)

            kND.fit(all_other_metafeatures, runs)
            self.kND = kND
        return self.kND.kBestSuggestions(dataset_metafeatures, k=-1,
            exclude_double_configurations=exclude_double_configurations)

    def _get_metafeatures(self):
        """This is inside an extra function for testing purpose"""
        # Load the task

        self.logger.info("Going to use the metafeature subset: %s", self.subset)
        all_metafeatures = self.meta_base.get_all_metafeatures()
        self.logger.info(" ".join(all_metafeatures.columns))

        # TODO: buggy and hacky, replace with a list seperated by commas
        if self.use_features and \
                (type(self.use_features) != str or self.use_features != ''):
            #ogger.warn("Going to keep the following features %s",
            #        str(self.use_features))
            if type(self.use_features) == str:
                use_features = self.use_features.split(",")
            elif type(self.use_features) in (list, np.ndarray):
                use_features = self.use_features
            else:
                raise NotImplementedError(type(self.use_features))

            if len(use_features) == 0:
                self.logger.info("You just tried to remove all metafeatures...")
            else:
                keep = [col for col in all_metafeatures.columns if col in use_features]
                if len(use_features) == 0:
                    self.logger.info("You just tried to remove all metafeatures...")
                else:
                    all_metafeatures = all_metafeatures.loc[:,keep]
                    self.logger.info("Going to keep the following metafeatures:")
                    self.logger.info(str(keep))

        return self._split_metafeature_array(self.dataset_name, all_metafeatures)

    def _split_metafeature_array(self, dataset_name, metafeatures):
        """Split the metafeature array into dataset metafeatures and all other.

        This is inside an extra function for testing purpose.
        """
        dataset_metafeatures = metafeatures.loc[dataset_name].copy()
        metafeatures = metafeatures[metafeatures.index != dataset_name]
        return dataset_metafeatures, metafeatures

    def read_task_list(self, fh):
        dataset_filenames = list()
        for line in fh:
            line = line.replace("\n", "")
            if line:
                dataset_filenames.append(line)
            else:
                raise ValueError("Blank lines in the task list are not "
                                 "supported.")
        return dataset_filenames

    def read_experiments_list(self, fh):
        experiments_list = list()
        for line in fh.readlines():
            experiments_list.append(line.split())
        return experiments_list


def parse_parameters(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task_file", type=str,
                        help="The task which should be optimized.")
    parser.add_argument("task_files_list", type=str,
                        help="A list with all task files for which "
                             "should be considered for metalearning")
    parser.add_argument("experiment_files_list", type=str,
                        help="A list with all experiment pickles which "
                             "should be considered for metalearning")
    parser.add_argument("metalearning_directory", type=str,
                        help="A directory with the metalearning datastructure")
    parser.add_argument("-d", "--distance_measure", type=str, default='l1',
                        choices=['l1', 'l2', 'learned', 'random', 'mfs_l1',
                                 'mfw_l1'])
    parser.add_argument("--metafeatures_subset", type=str, default='all',
                        choices=["pfahringer_2000_experiment1",
                                 "all", "yogotama_2014",
                                 "bardenet_2013_boost", "bardenet_2013_nn"])
    parser.add_argument("--distance_keep_features", type=str, default='',)
    parser.add_argument("--metric_kwargs", type=str, default='')
    parser.add_argument("--cli_target")
    # parser.add_argument("-p", "--params", required=True)
    parser.add_argument("--cwd", type=str)
    parser.add_argument("--number_of_jobs", required=True, type=int,
                        default=50)
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args(args=args)
    return args


"""
def main():
    args = parse_parameters()
    if args.cwd:
        os.chdir(args.cwd)

    cli_function = optimizer_base.command_line_function
    fn = functools.partial(cli_function, args.cli_target)

    with open(args.task_files_list) as fh:
         task_filenames = fh.readlines()
    with open(args.experiment_files_list) as fh:
        experiment_filenames = fh.readlines()

    if args.metric_kwargs:
        metric_kwargs = ast.literal_eval(args.metric_kwargs)
    else:
        metric_kwargs = None

    optimizer = MetaLearningOptimizer(args.task_file, task_filenames,
        experiment_filenames, args.cwd, distance=args.distance_measure,
        seed=args.seed, use_features=args.distance_keep_features,
        metric_kwargs=metric_kwargs, subset=args.metafeatures_subset)
    try:
        optimizer.perform_sequential_optimization(
            target_algorithm=fn,
            evaluation_budget=args.number_of_jobs)
    except StopIteration:
        logger.warning("No more hyperparameter configurations to be chosen "
                       "via metalearning!")


if __name__ == "__main__":
    main()
    exit(0)
"""