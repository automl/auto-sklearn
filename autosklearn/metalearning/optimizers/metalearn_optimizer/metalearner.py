import ast

import pandas as pd
import numpy as np
import sklearn.utils

from autosklearn.metalearning.metalearning.kNearestDatasets.kND import KNearestDatasets

from ....util.logging_ import get_logger


def test_function(params):
    return np.random.random()


class MetaLearningOptimizer(object):
    def __init__(self, dataset_name, configuration_space,
                 meta_base, distance='l1', seed=None, use_features=None,
                 distance_kwargs=None):
        self.dataset_name = dataset_name
        self.configuration_space = configuration_space
        self.meta_base = meta_base
        self.distance = distance
        self.seed = seed
        self.use_features = use_features
        self.distance_kwargs = distance_kwargs
        self.kND = None     # For caching, makes things faster...

        self.logger = get_logger(__name__)

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
        dataset_metafeatures, all_other_metafeatures = \
            self._split_metafeature_array()

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

    def _split_metafeature_array(self):
        dataset_metafeatures = self.meta_base.get_metafeatures(
            self.dataset_name, self.use_features)
        all_other_datasets = self.meta_base.get_all_dataset_names()
        all_other_datasets.remove(self.dataset_name)
        all_other_metafeatures = self.meta_base.get_metafeatures(
            all_other_datasets, self.use_features)
        return dataset_metafeatures, all_other_metafeatures
