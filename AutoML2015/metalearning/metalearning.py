import os
from StringIO import StringIO
import time

import numpy as np

from HPOlibConfigSpace.hyperparameters import InactiveHyperparameter

import pyMetaLearn.metafeatures.metafeatures as metafeatures
import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as \
    metalearner


class MetaLearning(object):
    """Right now, pyMetaLearn performs a OneHotEncoding if necessary, but it
    is really not necessary. This object helps to circumvent this by:

    1. call metafeatures.calculate_all_metafeatures() only for the
        metafeatures which do not need OneHotEncoded data
    2. Allows the caller to then perform a OneHotEncoding
    3. call metafeatures.calculate_metafeatures_encoded_labels() for all
        other metafeatures need OneHotEncoded data.
    """

    def __init__(self):
        self._metafeatures_encoded_labels = None
        self._metafeatures_labels = None
        # Hard-coded list of too-expensive metafeatures!
        self._exclude_metafeatures = set(['landmark_1NN',
                                          'landmark_decision_node_learner',
                                          'landmark_decision_tree',
                                          'landmark_lda'])

    def calculate_metafeatures_with_labels(self, X_train, Y_train,
                                           categorical, dataset_name):
        if self._metafeatures_labels is not None:
            raise ValueError("This method was already called!")

        self._metafeatures_labels = metafeatures. \
            calculate_all_metafeatures_with_labels(
            X_train, Y_train, categorical, dataset_name,
            dont_calculate=self._exclude_metafeatures)

    def calculate_metafeatures_encoded_labels(self, X_train, Y_train,
                                              categorical, dataset_name):
        if self._metafeatures_encoded_labels is not None:
            raise ValueError("This method was already called!")

        if np.sum(categorical) != 0:
            raise ValueError("Training matrix doesn't look OneHotEncoded!")

        self._metafeatures_encoded_labels = metafeatures.\
            calculate_all_metafeatures_encoded_labels(
            X_train, Y_train, categorical, dataset_name,
            dont_calculate=self._exclude_metafeatures)

    def create_metalearning_string_for_smac_call(self, configuration_space,
                                                 dataset_name, metric):
        if self._metafeatures_encoded_labels is None or \
                self._metafeatures_labels is None:
            raise ValueError("Please call "
                             "calculate_metafeatures_encoded_labels and "
                             "calculate_metafeatures_with_labels first!")

        current_directory = os.path.dirname(__file__)
        datasets_file = os.path.join(current_directory, "files", "datasets.yaml")
        experiments_file = os.path.join(current_directory, "files",
                                        "%s.experiments.yaml" % metric)

        # Concatenate the metafeatures!
        mf = self._metafeatures_labels
        mf.metafeature_values.extend(
            self._metafeatures_encoded_labels.metafeature_values)
        self.mf = mf

        if not all([np.isfinite(mf.value)
                    for mf in self.mf.metafeature_values
                    if isinstance(mf.value, float)]):
            print "%s contains non-finite metafeatures!" % self.mf
            return []

        metafeatures_subset = metafeatures.subsets["all"]
        metafeatures_subset.difference_update(self._exclude_metafeatures)
        metafeatures_subset = list(metafeatures_subset)

        # TODO maybe replace by kND directly to remove unavailable configurations
        start = time.time()
        ml = metalearner.MetaLearningOptimizer(dataset_name, configuration_space,
            datasets_file, experiments_file, distance="l1", seed=1,
            use_features=metafeatures_subset, subset='all')
        print "Reading meta-data took %5.2f seconds" % (time.time() - start)

        # TODO This is hacky, I must find a different way of adding a new dataset!
        ml.meta_base.add_dataset_with_metafeatures(dataset_name, None, self.mf)
        runs = ml.metalearning_suggest_all(
            exclude_double_configurations=True)

        # = Convert these configurations into the SMAC CLI configuration format
        smac_initial_configuration_strings = []

        for run in runs[:25]:
            smac_initial_configuration_strings.append(
                self.convert_configuration_to_smac_string(run.configuration))

        return smac_initial_configuration_strings


    def convert_configuration_to_smac_string(self, configuration):
        """Convert configuration to string for SMAC option --initialChallengers.

        The expected format looks like this:

        .. code:: bash

            --initialChallengers "-alpha 1 -rho 1 -ps 0.1 -wp 0.00"
        """
        config_string = StringIO()
        config_string.write("--initial_challengers")

        for hyperparameter in configuration:
            if isinstance(hyperparameter, InactiveHyperparameter):
                continue
            name = hyperparameter.hyperparameter.name
            config_string.write(" -%s %s" % (name, configuration[name].value))

        return config_string.getvalue()