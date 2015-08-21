# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
from __future__ import print_function

import os
import time
from StringIO import StringIO

import numpy as np
from pyMetaLearn.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, \
    calculate_all_metafeatures_encoded_labels, subsets

from pyMetaLearn.optimizers.metalearn_optimizer.metalearner import \
    MetaLearningOptimizer

from autosklearn.constants import *
from autosklearn.util import get_logger

logger = get_logger(os.path.basename(__file__))

# todo
# переписать этот код без класса
SENTINEL = 'uiaeo'

EXCLUDE_META_FUTURES = set([
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC'
])


def calculate_metafeatures_with_labels(X_train, Y_train, categorical,
                                       dataset_name):
    return calculate_all_metafeatures_with_labels(
        X_train, Y_train, categorical, dataset_name + SENTINEL,
        dont_calculate=EXCLUDE_META_FUTURES)


def calculate_metafeatures_encoded_labels(X_train, Y_train,
                                          categorical, dataset_name):
    if np.sum(categorical) != 0:
        raise ValueError("Training matrix doesn't look OneHotEncoded!")

    return calculate_all_metafeatures_encoded_labels(
        X_train, Y_train, categorical, dataset_name + SENTINEL,
        dont_calculate=EXCLUDE_META_FUTURES)


def convert_configuration_to_smac_string(configuration):
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


def create_metalearning_string_for_smac_call(
        metafeatures_encoded_labels, metafeatures_labels,
                                             configuration_space,
                                             dataset_name,
                                             metric,
                                             task,
                                             sparse,
                                             num_initial_configurations,
                                             metadata_directory):
    task = TASK_TYPES_TO_STRING[task]

    if metafeatures_encoded_labels is None or \
                    self._metafeatures_labels is None:
        raise ValueError('Please call '
                         'calculate_metafeatures_encoded_labels and '
                         'calculate_metafeatures_with_labels first!')

    current_directory = os.path.dirname(__file__)
    if metadata_directory is None:
        metadata_directory = os.path.join(current_directory, 'files',
                                          '%s_%s_%s' % (task, 'sparse' if
                                          sparse is True else
                                          'dense', metric))

    # Concatenate the metafeatures!
    mf = metafeatures_labels
    mf.metafeature_values.update(
        metafeatures_encoded_labels.metafeature_values)


    metafeatures_subset = subsets['all']
    metafeatures_subset.difference_update(EXCLUDE_META_FUTURES)
    metafeatures_subset = list(metafeatures_subset)

    start = time.time()
    ml = MetaLearningOptimizer(
        dataset_name=dataset_name + SENTINEL,
        configuration_space=configuration_space,
        aslib_directory=metadata_directory,
        distance='l1',
        seed=1,
        use_features=metafeatures_subset,
        subset='all')
    logger.info('Reading meta-data took %5.2f seconds',
                time.time() - start)
    print(ml.meta_base.configurations.keys())
    # TODO This is hacky, I must find a different way of adding a new
    # dataset!
    ml.meta_base.add_dataset(dataset_name + SENTINEL, self.mf)
    runs = ml.metalearning_suggest_all(exclude_double_configurations=True)

    # = Convert these configurations into the SMAC CLI configuration format
    smac_initial_configuration_strings = []

    for configuration in runs[:num_initial_configurations]:
        smac_initial_configuration_strings.append(
            convert_configuration_to_smac_string(configuration))

    return smac_initial_configuration_strings


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
        self._sentinel = 'uiaeo'
        self._metafeatures_encoded_labels = None
        self._metafeatures_labels = None
        # Hard-coded list of too-expensive metafeatures!
        self._exclude_metafeatures = set(
            ['Landmark1NN', 'LandmarkDecisionNodeLearner',
             'LandmarkDecisionTree', 'LandmarkLDA', 'LandmarkNaiveBayes',
             'PCAFractionOfComponentsFor95PercentVariance',
             'PCAKurtosisFirstPC', 'PCASkewnessFirstPC'])

    @property
    def metafeatures_labels(self):
        return self._metafeatures_labels

    @metafeatures_labels.setter
    def metafeatures_labels(self, value):
        self._metafeatures_labels = value

    @property
    def metafeatures_encoded_labels(self):
        return self._metafeatures_encoded_labels

    @metafeatures_encoded_labels.setter
    def metafeatures_encoded_labels(self, value):
        self._metafeatures_encoded_labels = value

    def calculate_metafeatures_with_labels(self, X_train, Y_train, categorical,
                                           dataset_name):
        if self._metafeatures_labels is not None:
            raise ValueError('This method was already called!')

        self._metafeatures_labels = metafeatures. \
            calculate_all_metafeatures_with_labels(
                X_train, Y_train, categorical, dataset_name + self._sentinel,
                dont_calculate=self._exclude_metafeatures)

    def calculate_metafeatures_encoded_labels(self, X_train, Y_train,
                                              categorical, dataset_name):
        if self._metafeatures_encoded_labels is not None:
            raise ValueError('This method was already called!')

        if np.sum(categorical) != 0:
            raise ValueError("Training matrix doesn't look OneHotEncoded!")

        self._metafeatures_encoded_labels = metafeatures. \
            calculate_all_metafeatures_encoded_labels(
                X_train, Y_train, categorical, dataset_name + self._sentinel,
                dont_calculate=self._exclude_metafeatures)

    def create_metalearning_string_for_smac_call(
            self, configuration_space, dataset_name, metric, task, sparse,
            num_initial_configurations, metadata_directory):
        task = TASK_TYPES_TO_STRING[task]

        if self._metafeatures_encoded_labels is None or \
                self._metafeatures_labels is None:
            raise ValueError('Please call '
                             'calculate_metafeatures_encoded_labels and '
                             'calculate_metafeatures_with_labels first!')

        current_directory = os.path.dirname(__file__)
        if metadata_directory is None:
            metadata_directory = os.path.join(current_directory, 'files',
                                              '%s_%s_%s' % (task, 'sparse' if
                                                            sparse is True else
                                                            'dense', metric))

        # Concatenate the metafeatures!
        mf = self._metafeatures_labels
        mf.metafeature_values.update(
            self._metafeatures_encoded_labels.metafeature_values)
        self.mf = mf

        metafeatures_subset = metafeatures.subsets['all']
        metafeatures_subset.difference_update(self._exclude_metafeatures)
        metafeatures_subset = list(metafeatures_subset)

        start = time.time()
        ml = MetaLearningOptimizer(
            dataset_name=dataset_name + self._sentinel,
            configuration_space=configuration_space,
            aslib_directory=metadata_directory,
            distance='l1',
            seed=1,
            use_features=metafeatures_subset,
            subset='all')
        logger.info('Reading meta-data took %5.2f seconds',
                    time.time() - start)
        print(ml.meta_base.configurations.keys())
        # TODO This is hacky, I must find a different way of adding a new
        # dataset!
        ml.meta_base.add_dataset(dataset_name + self._sentinel, self.mf)
        runs = ml.metalearning_suggest_all(exclude_double_configurations=True)

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
