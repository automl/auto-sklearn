import os

from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.arff_data_manager import ARFFDataManager

from autosklearn.constants import *



def get_data_manager(namespace, encode_labels=False):
    """Get a dataset from a argparse.Namespace.

    Parameters
    ----------
    namespace : argparse.Namespace
        The return value of ArgumentParser.parse_args()

    encode_labels : bool (default=False)
        Whether to perform 1HotEncoding for Categorical data.

    Returns
    -------
    DataManager
        Loaded dataset.
    """
    data_format = namespace.data_format
    data_format_ = data_format.lower()
    data_format_ = data_format_.replace('-', '').replace('_', '')

    if data_format_ == 'arff':
        return ARFFDataManager(namespace.dataset, namespace.task,
                               namespace.metric, namespace.target,
                               encode_labels=encode_labels)
    elif data_format_ == 'automlcompetitionformat':
        return CompetitionDataManager(namespace.dataset,
                                      encode_labels=encode_labels)
    else:
        raise NotImplementedError('Data format %s not supported.' % data_format)


def populate_argparse_with_data_options(parser):
    # Support different parsers for different data types

    parser.add_argument('--data-format', type=str, required=True,
                        choices=["automl-competition-format", "arff"])
    parser.add_argument('--dataset', type=str, required=True,
                        help='Unique identifier of the dataset.')
    parser.add_argument('--task', choices=TASK_TYPES_TO_STRING.values(),
                        help='Task to execute on the dataset. Only necessary '
                             'for data in arff format.')
    parser.add_argument('--metric', choices=list(STRING_TO_METRIC.keys()),
                        help='Loss function to optimize for. Only necessary '
                             'for data in arff format.')
    parser.add_argument('--target', type=str,
                        help='Target attribute. Only necessary for data in '
                             'arff format.')

    return parser
