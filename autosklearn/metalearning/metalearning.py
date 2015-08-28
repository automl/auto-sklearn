# -*- encoding: utf-8 -*-
"""Right now, pyMetaLearn performs a OneHotEncoding if necessary, but it
is really not necessary. This object helps to circumvent this by:

1. call metafeatures.calculate_all_metafeatures() only for the
    metafeatures which do not need OneHotEncoded data
2. Allows the caller to then perform a OneHotEncoding
3. call metafeatures.calculate_metafeatures_encoded_labels() for all
    other metafeatures need OneHotEncoded data.
"""
from __future__ import print_function
from autosklearn.conf import META_LEARNING_FOLDER

__all__ = [
    'calc_meta_features',
    'calc_meta_features_encoded',
    'convert_conf2smac_string',
    'create_metalearning_string_for_smac_call',
]


from autosklearn.constants import TASK_TYPES_TO_STRING

import os
import time
from StringIO import StringIO

import numpy as np
from pyMetaLearn.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, \
    calculate_all_metafeatures_encoded_labels, subsets

from pyMetaLearn.optimizers.metalearn_optimizer.metalearner import \
    MetaLearningOptimizer


from autosklearn.util import get_logger

logger = get_logger(os.path.basename(__file__))

SENTINEL = 'uiaeo'

EXCLUDE_META_FUTURES = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC'
}

def calc_meta_features(X_train, Y_train, categorical, dataset_name):
    """
    Calculate meta features with label
    :param X_train:
    :param Y_train:
    :param categorical:
    :param dataset_name:
    :return:
    """
    return calculate_all_metafeatures_with_labels(
        X_train, Y_train, categorical, dataset_name + SENTINEL,
        dont_calculate=EXCLUDE_META_FUTURES)


def calc_meta_features_encoded(X_train, Y_train, categorical, dataset_name):
    """
    Calculate meta features with encoded labels
    :param X_train:
    :param Y_train:
    :param categorical:
    :param dataset_name:
    :return:
    """
    if np.sum(categorical) != 0:
        raise ValueError("Training matrix doesn't look OneHotEncoded!")

    return calculate_all_metafeatures_encoded_labels(
        X_train, Y_train, categorical, dataset_name + SENTINEL,
        dont_calculate=EXCLUDE_META_FUTURES)


def convert_conf2smac_string(configuration):
    """
    Convert configuration to string for SMAC option --initialChallengers.

    The expected format looks like this:

    .. code:: bash

        --initialChallengers "-alpha 1 -rho 1 -ps 0.1 -wp 0.00"
    :param configuration:
    :return:
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
        metafeatures_labels,
        metafeatures_encoded_labels,
        configuration_space,
        dataset_name,
        metric,
        task,
        sparse,
        num_initial_configurations,
        metadata_directory):
    """

    :param metafeatures_labels:
    :param metafeatures_encoded_labels:
    :param configuration_space:
    :param dataset_name:
    :param metric:
    :param task:
    :param sparse:
    :param num_initial_configurations:
    :param metadata_directory:
    :return:
    """
    task = TASK_TYPES_TO_STRING[task]

    if metafeatures_encoded_labels is None or \
                    metafeatures_labels is None:
        raise ValueError('Please call '
                         'calculate_metafeatures_encoded_labels and '
                         'calculate_metafeatures_with_labels first!')

    if metadata_directory is None:
        metadata_directory = os.path.join(META_LEARNING_FOLDER,
                                          '%s_%s_%s' % (task, 'sparse' if
                                          sparse is True else
                                          'dense', metric))

    # todo
    # Установка эксепшена здесь не прерывает работу всей программы
    # найти где пропускается эксепншн

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
    ml.meta_base.add_dataset(dataset_name + SENTINEL, mf)
    runs = ml.metalearning_suggest_all(exclude_double_configurations=True)

    # = Convert these configurations into the SMAC CLI configuration format
    smac_initial_configuration_strings = []

    for configuration in runs[:num_initial_configurations]:
        smac_initial_configuration_strings.append(
            convert_conf2smac_string(configuration))

    return smac_initial_configuration_strings
