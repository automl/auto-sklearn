import os
from StringIO import StringIO
import time

import numpy as np

import pyMetaLearn.metafeatures.metafeatures as metafeatures
import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as \
    metalearner

from autosklearn.util import logging_
logger = logging_.get_logger(__name__)


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
        self._sentinel = "uiaeo"
        self._metafeatures_encoded_labels = None
        self._metafeatures_labels = None
        # Hard-coded list of too-expensive metafeatures!
        self._exclude_metafeatures = set(['Landmark1NN',
                                          'LandmarkDecisionNodeLearner',
                                          'LandmarkDecisionTree',
                                          'LandmarkLDA',
                                          'LandmarkNaiveBayes',
                                          'PCAFractionOfComponentsFor95PercentVariance',
                                          'PCAKurtosisFirstPC',
                                          'PCASkewnessFirstPC'])

    def calculate_metafeatures_with_labels(self, X_train, Y_train,
                                           categorical, dataset_name):
        if self._metafeatures_labels is not None:
            raise ValueError("This method was already called!")

        self._metafeatures_labels = metafeatures. \
            calculate_all_metafeatures_with_labels(
            X_train, Y_train, categorical, dataset_name + self._sentinel,
            dont_calculate=self._exclude_metafeatures)

    def calculate_metafeatures_encoded_labels(self, X_train, Y_train,
                                              categorical, dataset_name):
        if self._metafeatures_encoded_labels is not None:
            raise ValueError("This method was already called!")

        if np.sum(categorical) != 0:
            raise ValueError("Training matrix doesn't look OneHotEncoded!")

        self._metafeatures_encoded_labels = metafeatures.\
            calculate_all_metafeatures_encoded_labels(
            X_train, Y_train, categorical, dataset_name + self._sentinel,
            dont_calculate=self._exclude_metafeatures)

    def create_metalearning_string_for_smac_call(self, configuration_space,
            dataset_name, metric, task, sparse, num_initial_configurations,
            metadata_directory):
        if self._metafeatures_encoded_labels is None or \
                self._metafeatures_labels is None:
            raise ValueError("Please call "
                             "calculate_metafeatures_encoded_labels and "
                             "calculate_metafeatures_with_labels first!")

        current_directory = os.path.dirname(__file__)
        if metadata_directory is None:
            metadata_directory = os.path.join(current_directory, "files",
                "%s_%s_%s" %(task, "sparse" if sparse is True else "dense", metric))

        # Concatenate the metafeatures!
        mf = self._metafeatures_labels
        mf.metafeature_values.update(
            self._metafeatures_encoded_labels.metafeature_values)
        self.mf = mf

        metafeatures_subset = metafeatures.subsets["all"]
        metafeatures_subset.difference_update(self._exclude_metafeatures)
        metafeatures_subset = list(metafeatures_subset)

        start = time.time()
        ml = metalearner.MetaLearningOptimizer(
            dataset_name=dataset_name + self._sentinel,
            configuration_space=configuration_space,
            aslib_directory=metadata_directory, distance="l1",
            seed=1, use_features=metafeatures_subset, subset='all')
        logger.info("Reading meta-data took %5.2f seconds",
                    time.time() - start)
        print ml.meta_base.configurations.keys()
        # TODO This is hacky, I must find a different way of adding a new dataset!
        ml.meta_base.add_dataset(dataset_name + self._sentinel, self.mf)
        runs = ml.metalearning_suggest_all(
            exclude_double_configurations=True)

        # = Convert these configurations into the SMAC CLI configuration format
        smac_initial_configuration_strings = []

        for configuration in runs[:num_initial_configurations]:
            smac_initial_configuration_strings.append(
                self.convert_configuration_to_smac_string(configuration))

        return smac_initial_configuration_strings



    def convert_configuration_to_smac_string(self, configuration):
        """Convert configuration to string for SMAC option --initialChallengers.

        The expected format looks like this:

        .. code:: bash

            --initialChallengers "-alpha 1 -rho 1 -ps 0.1 -wp 0.00"
        """
        config_string = StringIO()
        config_string.write("--initial-challengers \"")

        for hp_name in sorted(configuration):
            value = configuration[hp_name]
            if value is None:
                continue
            config_string.write(" -%s '%s'" % (hp_name, value))

        config_string.write("\"")
        return config_string.getvalue()