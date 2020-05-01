import pandas as pd

from ...util.logging_ import get_logger
from ..input import aslib_simple
from ..metafeatures.metafeature import DatasetMetafeatures
from ConfigSpace.configuration_space import Configuration


class Run(object):
    def __init__(self, configuration, result, runtime):
        self.configuration = configuration
        self.result = result
        self.runtime = runtime

    def __repr__(self):
        return "Run:\nresult: %3.3f\nruntime: %3.3f\n%s" % \
               (self.result, self.runtime, str(self.configuration))


class Instance(object):
    def __init__(self, name, features):
        self.name = name
        self.features = features


class MetaBase(object):
    def __init__(self, configuration_space, aslib_directory):
        """Container for dataset metadata and experiment results.

        Constructor arguments:
        - The configuration space
        - aslib_directory: directory with a problem instance in the aslib format
        """

        self.logger = get_logger(__name__)

        self.configuration_space = configuration_space
        self.aslib_directory = aslib_directory

        aslib_reader = aslib_simple.AlgorithmSelectionProblem(self.aslib_directory)
        self.metafeatures = aslib_reader.metafeatures
        self.algorithm_runs = aslib_reader.algorithm_runs
        self.configurations = aslib_reader.configurations

        configurations = dict()
        for algorithm_id in self.configurations:
            configuration = self.configurations[algorithm_id]
            try:
                configurations[str(algorithm_id)] = \
                    (Configuration(configuration_space, values=configuration))
            except (ValueError, KeyError) as e:
                self.logger.debug("Error reading configurations: %s", e)

        self.configurations = configurations

    def add_dataset(self, name, metafeatures):
        metafeatures.name = name
        if isinstance(metafeatures, DatasetMetafeatures):
            data_ = {mf.name: mf.value for mf in metafeatures.metafeature_values.values()}
            metafeatures = pd.Series(name=name, data=data_)
        if name in self.metafeatures.index:
            self.logger.warning(
                'Dataset %s already in meta-data. Removing occurence.', name
            )
            self.metafeatures.drop(name, inplace=True)
        self.metafeatures = self.metafeatures.append(metafeatures)

        runs = pd.Series([], name=name)
        for metric in self.algorithm_runs.keys():
            self.algorithm_runs[metric].append(runs)

    def get_runs(self, dataset_name, performance_measure=None):
        """Return a list of all runs for a dataset."""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure].loc[dataset_name]

    def get_all_runs(self, performance_measure=None):
        """Return a dictionary with a list of all runs"""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure]

    def get_metafeatures(self, dataset_name=None, features=None):
        if features is not None:
            metafeatures = self._get_metafeatures(features)
        else:
            metafeatures = self.metafeatures
        if dataset_name is not None:
            return metafeatures.loc[dataset_name]
        else:
            return metafeatures

    def _get_metafeatures(self, features):
        """This is inside an extra function for testing purpose"""
        # Load the task

        self.logger.info("Going to use the following metafeature subset: %s",
                         features)
        all_metafeatures = self.metafeatures
        all_metafeatures = all_metafeatures.loc[:, features]

        return all_metafeatures

    def get_configuration_from_algorithm_index(self, idx):
        return self.configurations[str(idx)]
        # configuration = self.configurations[idx]
        # configuration = Configuration(self.configuration_space,
        # **configuration)
        # return configuration

    def get_algorithm_index_from_configuration(self, configuration):
        for idx in self.configurations.keys():
            if configuration == self.configurations[idx]:
                return idx

        raise ValueError(configuration)

    def get_all_dataset_names(self):
        return list(self.metafeatures.index)
