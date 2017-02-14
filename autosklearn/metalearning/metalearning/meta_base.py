import logging

import pandas as pd

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

        self.logger = logging.getLogger(__name__)

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
                configurations[algorithm_id] = \
                    (Configuration(configuration_space, values=configuration))
            except (ValueError, KeyError) as e:
                self.logger.debug("Error reading configurations: %s", e)

        self.configurations = configurations

    def add_dataset(self, name, metafeatures):
        metafeatures.name = name
        if isinstance(metafeatures, DatasetMetafeatures):
            metafeatures = pd.Series(name=name,
                data={mf.name: mf.value for mf in
                      metafeatures.metafeature_values.values()})
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

    def get_metafeatures(self, dataset_name):
        dataset_metafeatures = self.metafeatures.loc[dataset_name]
        return dataset_metafeatures

    def get_all_metafeatures(self):
        """Create a pandas DataFrame for the metadata of all datasets."""
        return self.metafeatures

    def get_configuration_from_algorithm_index(self, idx):
        return self.configurations[idx]
        #configuration = self.configurations[idx]
        #configuration = Configuration(self.configuration_space,
        # **configuration)
        #return configuration

    def get_algorithm_index_from_configuration(self, configuration):
        for idx in self.configurations.keys():
            if configuration == self.configurations[idx]:
                return idx

        raise ValueError(configuration)
