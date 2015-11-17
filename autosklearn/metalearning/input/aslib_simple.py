from collections import defaultdict, OrderedDict
import csv
import logging
import os

import arff
import pandas as pd


class AlgorithmSelectionProblem(object):
    def __init__(self, directory):
        self.logger = logging.getLogger(__name__)

        # Create data structures
        self.dir_ = directory
        self.algorithm_runs = None
        self.configurations = None
        self.metafeatures = None
        self.read_funcs = {
            # "description.txt": self._read_description,
            "algorithm_runs.arff": self._read_algorithm_runs,
            # "feature_costs.arff": self._read_feature_costs,
            "feature_values.arff": self._read_feature_values,
            #"feature_runstatus.arff": self._read_feature_runstatus,
            #"ground_truth.arff": self._read_ground_truth,
            #"cv.arff": self._read_cv,
            "configurations.csv": self._read_configurations
        }
        self.found_files = []

        # Read ASLib files
        self._find_files()
        self._read_files()

    def _find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = [#"description.txt",
                    "algorithm_runs.arff",
                    "feature_values.arff",]
                    #"feature_runstatus.arff"]
        optional = ["ground_truth.arff", "feature_costs.arff", "citation.bib",
                    "cv.arff", "configurations.csv"]

        for expected_file in expected:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.error(
                    "Not found: %s (has to be added)" % (full_path))
            else:
                self.found_files.append(full_path)

        for expected_file in optional:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.warning(
                    "Not found: %s (maybe you want to add it)" % (full_path))
            else:
                self.found_files.append(full_path)

    def _read_files(self):
        '''
            iterates over all found files (self.found_files) and
            calls the corresponding function to validate file
        '''
        for file_ in self.found_files:
            read_func = self.read_funcs.get(os.path.basename(file_))
            if read_func:
                read_func(file_)

    def _read_algorithm_runs(self, filename):
        with open(filename) as fh:
            arff_dict = arff.load(fh)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (filename))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (filename))
        if arff_dict["attributes"][2][0].upper() != "ALGORITHM":
            self.logger.error(
                "algorithm as third attribute is missing in %s" % (filename))

        performance_measures = [pm[0] for pm in arff_dict['attributes'][3:-1]]

        measure_instance_algorithm_triples = defaultdict(lambda: defaultdict(dict))
        for data in arff_dict["data"]:
            inst_name = str(data[0])
            repetition = data[1]
            algorithm = str(data[2])
            perf_list = data[3:-1]
            status = data[-1]

            for i, performance_measure in enumerate(performance_measures):
                measure_instance_algorithm_triples[performance_measure][
                    inst_name][algorithm] = perf_list[i]

        # TODO: this does not support any repetitions!
        measure_algorithm_matrices = OrderedDict()
        for pm in performance_measures:
            measure_algorithm_matrices[pm] = pd.DataFrame(
                measure_instance_algorithm_triples[pm]).transpose()

        self.algorithm_runs = measure_algorithm_matrices

    def _read_feature_values(self, filename):
        with open(filename) as fh:
            arff_dict = arff.load(fh)

        metafeatures = dict()
        for data in arff_dict["data"]:
            inst_name = data[0]
            repetition = data[1]
            features = data[2:]

            metafeatures[inst_name] = {feature[0]: feature_value
                for feature, feature_value in
                zip(arff_dict['attributes'][2:], features)}

        self.metafeatures = pd.DataFrame(metafeatures).transpose()

    def _read_configurations(self, filename):
        with open(filename) as fh:
            csv_reader = csv.DictReader(fh)

            configurations = dict()
            for line in csv_reader:
                configuration = dict()
                algorithm_id = line['idx']
                for hp_name, value in line.items():
                    if not value or hp_name == 'idx':
                        continue

                    try:
                        value = int(value)
                    except:
                        try:
                            value = float(value)
                        except:
                            pass

                    configuration[hp_name] = value
                configurations[algorithm_id] = configuration
        self.configurations = configurations
