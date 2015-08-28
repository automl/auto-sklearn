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
        dataset = namespace.dataset
        if dataset[-1] == '/':
            dataset = dataset[-1]
        dataset = os.path.split(dataset)
        dir = dataset[0]
        name = dataset[1]
        return CompetitionDataManager(name, dir,
                                      encode_labels=encode_labels)
    else:
        raise NotImplemented('Data format %s not supported.' % data_format)


def populate_argparse_with_data_options(parser):
    # Support different parsers for different data types
    subparsers = parser.add_subparsers(
        title='data',
        description='further arguments concerning how and which '
                    'data is loaded.',
        dest='data_format')

    arff_parser = subparsers.add_parser('arff')
    arff_parser.add_argument('--dataset', type=str, required=True,
                             help='Unique identifier of the dataset.')
    arff_parser.add_argument('--task', help='Task to execute on the dataset.',
                             choices=TASK_TYPES_TO_STRING.values(),
                             required=True)
    arff_parser.add_argument('--metric', help='Loss function to optimize for.',
                             choices=['acc_metric', 'auc_metric',
                                      'bac_metric', 'f1_metric', 'pac_metric'],
                             required=True)
    arff_parser.add_argument('--target', help='Target attribute.',
                             required=True)

    automl_competition_parser = subparsers.add_parser(
        'automl-competition-format')
    automl_competition_parser.add_argument(
        '--dataset', type=str, required=True,
        help='Unique identifier of the dataset.')

    return parser
