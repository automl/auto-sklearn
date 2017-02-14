'''
Created on Dez 10, 2013

@author: Marius Lindauer

Problems:
  * feature cost vs. feature cutoff time ... how to replace ?
'''
import logging
import os

import arff

logger = logging.getLogger(__name__)


class Metainfo(object):
    '''
        all meta information about an algorithm selection scenario
    '''


    def __init__(self):
        # listed in description.txt
        self.scenario = None  # string
        self.performance_measure = []  # list of strings
        self.performance_type = []  # list of "runtime" or "solution_quality"
        self.maximize = []  # list of "true" or "false"
        self.algorithm_cutoff_time = None  # float
        self.algorithm_cutoff_memory = None  # integer
        self.features_cutoff_time = None  # float
        self.features_cutoff_memory = None  # integer
        self.features_deterministic = []  # list of strings
        self.features_stochastic = []  # list of strings
        self.algorithms = []  # list of strings
        self.algortihms_deterministics = []  # list of strings
        self.algorithms_stochastic = []  # list of strings
        self.feature_group_dict = {}  # string -> [] of strings
        self.feature_steps = []

        # extracted in other files
        self.features = []
        self.ground_truths = {}  # type -> [values]
        self.cv_given = False

        # command line options
        self.options = None


class Instance(object):
    '''
       benchmark instance with runtime list, feature list and sat/unsat status
    '''


    def __init__(self, name):
        '''
        Constructor
        '''
        self._name = name
        self._cost = {}  # cost name -> algorithm -> [float]
        self._status = {}  # algorithm -> status
        self._cost_vec = []
        self._transformed_cost_vec = []
        self._features = []
        self._features_status = {}  # feature step -> status
        self._normed_features = None
        self._ground_truth = {}
        self._weight = 1
        self._feature_group_cost_dict = {}  # feature group -> cost
        self._feature_cost_total = 0.0  # float
        self._fold = {}  # mapping: repetition -> cv fold split
        self._pre_solved = False
        self._pre_solved_by_schedule = False

    def __str__(self):
        return self._name

    def get_name(self):
        return self._name

    def finished_input(self, algorithm_list):

        avg = lambda x: sum(x) / len(x)
        if "runtime" in self._cost.keys():
            cost_dict = self._cost["runtime"]
        elif "performance" in self._cost.keys():
            cost_dict = self._cost["performance"]
        else:
            logger.error(
                "Neither runtime nor performance is used as cost metric.\nClaspfolio does not support these data sets.")
        self._cost_vec = [avg(cost_dict[algo]) for algo in
                          algorithm_list]  # ensure the same order in all vectors
        self._transformed_cost_vec = [avg(cost_dict[algo]) for algo in
                                      algorithm_list]  # ensure the same order in all vectors

    def penalize_weight(self, gamma):
        '''
            new weight: weight * gamma
        '''
        self._weight *= gamma


class CosealReader(object):
    def __init__(self):
        '''
            Constructor
        '''

        self.dir_ = None  # directory with all files for parsing
        self.found_files = []  # all found files in self.dir_
        self.read_funcs = {
            "description.txt": self.read_description,
            "algorithm_runs.arff": self.read_algorithm_runs,
            "feature_costs.arff": self.read_feature_costs,
            "feature_values.arff": self.read_feature_values,
            "feature_runstatus.arff": self.read_feature_runstatus,
            "ground_truth.arff": self.read_ground_truth,
            "cv.arff": self.read_cv
        }

        self.metainfo = Metainfo()
        self.instances = {}


    def parse_coseal(self, coseal_dir, args_):
        '''
            main method of Checker
        '''
        # add command line arguments in metainfo
        self.metainfo.options = args_
        self.dir_ = coseal_dir
        self.find_files()
        self.read_files()
        self.check_feature_status()
        self.remove_features()
        self.check_instances()

        #empty algorithm dict
        algo_dict = dict((algo, "") for algo in self.metainfo.algorithms)

        if self.metainfo.options.feat_time == -1:
            self.metainfo.options.feat_time = int(
                self.metainfo.algorithm_cutoff_time) / 10

        return self.instances, self.metainfo, algo_dict

    def find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = ["description.txt", "algorithm_runs.arff",
                    "feature_values.arff", "feature_runstatus.arff"]
        optional = ["ground_truth.arff", "feature_costs.arff", "citation.bib",
                    "cv.arff"]

        for expected_file in expected:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                logger.error("Not found: %s (has to be added)" % (full_path))
            else:
                self.found_files.append(full_path)

        for expected_file in optional:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                logger.warning(
                    "Not found: %s (maybe you want to add it)" % (full_path))
            else:
                self.found_files.append(full_path)

    def read_files(self):
        '''
            iterates over all found files (self.found_files) and 
            calls the corresponding function to validate file
        '''
        for file_ in self.found_files:
            read_func = self.read_funcs.get(os.path.basename(file_))
            if read_func:
                read_func(file_)

    def read_description(self, file_):
        '''
            reads description file
            and saves all meta information
        '''
        logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            for line in fp:
                line = line.replace("\n", "").strip(" ")
                if line.upper().startswith("SCENARIO_ID"):
                    self.metainfo.scenario = line.split(":")[1].strip(" ")
                elif line.upper().startswith("PERFORMANCE_MEASURES"):
                    self.metainfo.performance_measure = map(
                        lambda x: x.strip(" "),
                        line.split(":")[1].strip(" ").split(","))
                elif line.upper().startswith("MAXIMIZE"):
                    try:
                        self.metainfo.maximize = bool(
                            line.split(":")[1].strip(" ").split(","))
                    except ValueError:
                        logger.warning("Cannot read MAXIMIZE")
                elif line.upper().startswith("PERFORMANCE_TYPE"):
                    self.metainfo.performance_type = map(lambda x: x.strip(" "),
                                                         line.split(":")[
                                                             1].strip(
                                                             " ").split(","))
                elif line.upper().startswith("ALGORITHM_CUTOFF_TIME"):
                    try:
                        self.metainfo.algorithm_cutoff_time = float(
                            line.split(":")[1])
                    except ValueError:
                        logger.warning("Cannot read ALGORITHM_CUTOFF_TIME")
                elif line.upper().startswith("ALGORITHM_CUTOFF_MEMORY"):
                    try:
                        self.metainfo.algorithm_cutoff_memory = float(
                            line.split(":")[1])
                    except ValueError:
                        logger.warning("Cannot read ALGORITHM_CUTOFF_MEMORY")
                elif line.upper().startswith("FEATURES_CUTOFF_TIME"):
                    try:
                        self.metainfo.features_cutoff_time = float(
                            line.split(":")[1])
                    except ValueError:
                        logger.warning("Cannot read FEATURES_CUTOFF_TIME")
                elif line.upper().startswith("FEATURES_CUTOFF_MEMORY"):
                    try:
                        self.metainfo.features_cutoff_memory = float(
                            line.split(":")[1])
                    except ValueError:
                        logger.warning("Cannot read FEATURES_CUTOFF_MEMORY")
                elif line.upper().startswith("FEATURES_DETERMINISTIC"):
                    try:
                        self.metainfo.features_deterministic = map(
                            lambda x: x.strip(" "),
                            line.split(":")[1].strip(" ").split(","))
                    except ValueError:
                        logger.warning("Cannot read FEATURES_DETERMINISTIC")
                elif line.upper().startswith("FEATURES_STOCHASTIC"):
                    try:
                        self.metainfo.features_stochastic = map(
                            lambda x: x.strip(" "),
                            line.split(":")[1].strip(" ").split(","))
                    except ValueError:
                        logger.warning("Cannot read FEATURES_STOCHASTIC")
                elif line.upper().startswith("ALGORITHMS_DETERMINISTIC"):
                    try:
                        self.metainfo.algortihms_deterministics = filter(
                            lambda x: True if x else False,
                            map(lambda x: x.strip(" "),
                                line.split(":")[1].strip(" ").split(",")))
                    except ValueError:
                        logger.warning("Cannot read ALGORTIHMS_DETERMINISTIC")
                elif line.upper().startswith("ALGORITHMS_STOCHASTIC"):
                    try:
                        self.metainfo.algorithms_stochastic = filter(
                            lambda x: True if x else False,
                            map(lambda x: x.strip(" "),
                                line.split(":")[1].strip(" ").split(",")))
                    except ValueError:
                        logger.warning("Cannot read ALGORITHMS_STOCHASTIC")
                elif line.upper().startswith("FEATURE_STEP"):
                    try:
                        group_name = line.split(":")[0][12:].strip(" ")
                        features = map(lambda x: x.strip(" "),
                                       line.split(":")[1].strip(" ").split(","))
                        self.metainfo.feature_group_dict[group_name] = features
                    except ValueError:
                        logger.warning("Cannot read Feature_Step")
                elif line.startswith("default_step"):
                    try:
                        self.metainfo.feature_steps = filter(
                            lambda x: True if x else False,
                            map(lambda x: x.strip(" "),
                                line.split(":")[1].strip(" ").split(",")))
                    except ValueError:
                        logger.warning("Cannot read DEFAULT_STEPS")

        self.metainfo.algorithms = list(
            set(self.metainfo.algorithms_stochastic).union(
                self.metainfo.algortihms_deterministics))

        if not self.metainfo.scenario:
            logger.warning("Have not found SCENARIO_ID")
        if not self.metainfo.performance_measure:
            logger.warning("Have not found PERFORMANCE_MEASURE")
        if not self.metainfo.performance_type:
            logger.warning("Have not found PERFORMANCE_TYPE")
        if not self.metainfo.maximize:
            logger.warning("Have not found MAXIMIZE")
        if not self.metainfo.algorithm_cutoff_time:
            logger.error("Have not found algorithm_cutoff_time")
        if not self.metainfo.algorithm_cutoff_memory:
            logger.warning("Have not found algorithm_cutoff_memory")
        if not self.metainfo.features_cutoff_time:
            logger.warning("Have not found features_cutoff_time")
            logger.warning(
                "Assumption FEATURES_CUTOFF_TIME == ALGORITHM_CUTOFF_TIME ")
            self.metainfo.features_cutoff_time = self.metainfo.algorithm_cutoff_time
        if not self.metainfo.features_cutoff_memory:
            logger.warning("Have not found features_cutoff_memory")
        if not self.metainfo.features_deterministic:
            logger.warning("Have not found features_deterministic")
        if not self.metainfo.features_stochastic:
            logger.warning("Have not found features_stochastic")
        if not self.metainfo.algortihms_deterministics:
            logger.warning("Have not found algortihms_deterministics")
        if not self.metainfo.algorithms_stochastic:
            logger.warning("Have not found algorithms_stochastic")
        if not self.metainfo.feature_group_dict:
            logger.warning("Have not found any feature step")

        feature_intersec = set(
            self.metainfo.features_deterministic).intersection(
            self.metainfo.features_stochastic)
        if feature_intersec:
            logger.warning(
                "Intersection of deterministic and stochastic features is not empty: %s" % (
                    str(feature_intersec)))
        algo_intersec = set(
            self.metainfo.algortihms_deterministics).intersection(
            self.metainfo.algorithms_stochastic)
        if algo_intersec:
            logger.warning(
                "Intersection of deterministic and stochastic algorithms is not empty: %s" % (
                    str(algo_intersec)))


    def read_algorithm_runs(self, file_):
        '''
            read performance file
            and saves information
            add Instance() in self.instances
            
            unsuccessful runs are replaced by algorithm_cutoff_time if performance_type is runtime
            
            EXPECTED HEADER:
            @RELATION ALGORITHM_RUNS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE algorithm STRING
            @ATTRIBUTE PAR10 NUMERIC
            @ATTRIBUTE Number_of_satisfied_clauses NUMERIC
            @ATTRIBUTE runstatus {ok, timeout, memout, not_applicable, crash, other}
        '''
        logger.info("Read %s" % (file_))

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            logger.error(
                "repetition as second attribute is missing in %s" % (file_))
        if arff_dict["attributes"][2][0].upper() != "ALGORITHM":
            logger.error(
                "algorithm as third attribute is missing in %s" % (file_))

        i = 0
        for performance_measure in self.metainfo.performance_measure:
            if arff_dict["attributes"][3 + i][
                0].upper() != performance_measure.upper():
                logger.error("\"%s\" as attribute is missing in %s" % (
                    performance_measure, file_))
            i += 1

        if arff_dict["attributes"][3 + i][0].upper() != "RUNSTATUS":
            logger.error(
                "runstatus as last attribute is missing in %s" % (file_))

        pairs_inst_rep_alg = []
        for data in arff_dict["data"]:
            inst_name = str(data[0])
            repetition = data[1]
            algorithm = str(data[2])
            perf_list = data[3:-1]
            status = data[-1]

            inst_ = self.instances.get(inst_name, Instance(inst_name))

            for p_measure, p_type, perf in zip(
                    self.metainfo.performance_measure,
                    self.metainfo.performance_type, perf_list):
                if perf is None:
                    logger.error(
                        "The following performance data has missing values.\n" +
                        "%s" % (",".join(map(str, data))))
                if p_type == "runtime" and status.upper() != "OK":  # if broken run, replace with cutoff time
                    perf = self.metainfo.algorithm_cutoff_time + 1
                inst_._cost[p_measure] = inst_._cost.get(p_measure, {})
                perf_measure_dict = inst_._cost[p_measure]
                perf_measure_dict[algorithm] = perf_measure_dict.get(algorithm,
                    [])
                perf_measure_dict[algorithm].append(max(float(perf), 0.00001))

            inst_._status[algorithm] = status.upper()

            self.instances[inst_name] = inst_
            if (inst_name, repetition, algorithm) in pairs_inst_rep_alg:
                logger.warning("Pair (%s,%s,%s) is not unique in %s" % (
                    inst_name, repetition, algorithm, file_))
            else:
                pairs_inst_rep_alg.append((inst_name, repetition, algorithm))

    def read_feature_costs(self, file_):
        '''
            reads feature time file
            and saves in self.instances
            
            Expected header:
            @RELATION FEATURE_COSTS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing NUMERIC
            @ATTRIBUTE local_search_probing NUMERIC

        '''
        logger.info("Read %s" % (file_))

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "\"instance_id\" as first attribute is missing in %s" % (file_))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            logger.error(
                "\"repetition\" as second attribute is missing in %s" % (file_))
        found_groups = map(str, sorted(
            map(lambda x: x[0], arff_dict["attributes"][2:])))
        for meta_group in self.metainfo.feature_group_dict.keys():
            if meta_group not in found_groups:
                logger.error("\"%s\" as attribute is missing in %s" % (
                    meta_group, file_))

        pairs_inst_rep = []
        for data in arff_dict["data"]:
            inst_name = str(data[0])
            repetition = data[1]
            feature_cost = data[2:]

            inst_ = self.instances.get(inst_name)
            if not inst_:
                logger.warning(
                    "Instance \"%s\" has feature cost but was not found in algorithm_runs.arff" % (
                        inst_name))
                continue

            for cost, f_group in zip(feature_cost, arff_dict["attributes"][2:]):
                inst_._feature_group_cost_dict[str(f_group[0])] = cost

            if (inst_name, repetition) in pairs_inst_rep:
                logger.warning("Pair (%s,%s) is not unique in %s" % (
                    inst_name, repetition, file_))
            else:
                pairs_inst_rep.append((inst_name, repetition))

    def read_feature_values(self, file_):
        '''
            reads feature file
            and saves them in self.instances
            
            Expected Header:
            @RELATION FEATURE_VALUES_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE number_of_variables NUMERIC
            @ATTRIBUTE number_of_clauses NUMERIC
            @ATTRIBUTE first_local_min_steps NUMERIC
        '''

        logger.info("Read %s" % (file_))

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            logger.error(
                "repetition as second attribute is missing in %s" % (file_))

        feature_set = set(self.metainfo.features_deterministic).union(
            self.metainfo.features_stochastic)

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            self.metainfo.features.append(f_name)
            if not f_name in feature_set:
                logger.error(
                    "Feature \"%s\" was not defined as deterministic or stochastic" % (
                        f_name))

        pairs_inst_rep = []
        encoutered_features = []
        for data in arff_dict["data"]:
            inst_name = data[0]
            repetition = data[1]
            features = data[2:]

            if len(features) != len(self.metainfo.features):
                logger.error(
                    "Number of features in attributes does not match number of found features; instance: %s" % (
                        inst_name))

            if not self.instances.get(inst_name):
                logger.warning(
                    "Instance \"%s\" has features but was not found in performance file" % (
                        inst_name))
                continue

            inst_ = self.instances[inst_name]

            inst_._features = features  # TODO: handle feature repetitions

            # not only Nones in feature vector and previously seen
            if reduce(lambda x, y: True if (x or y) else False, features,
                      False) and features in encoutered_features:
                logger.warning("Feature vector found twice: %s" % (
                    ",".join(map(str, features))))
            else:
                encoutered_features.append(features)

            if (inst_name, repetition) in pairs_inst_rep:
                logger.warning("Pair (%s,%s) is not unique in %s" % (
                    inst_name, repetition, file_))
            else:
                pairs_inst_rep.append((inst_name, repetition))

    def read_feature_runstatus(self, file_):
        '''
            reads run stati of all pairs instance x feature step
            and saves them self.instances
            
            Expected header:
            @RELATION FEATURE_RUNSTATUS_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing { ok , timeout , memout , presolved , crash , other }
            @ATTRIBUTE local_search_probing { ok , timeout , memout , presolved , crash , other }
        '''
        logger.info("Read %s" % (file_))

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            logger.error(
                "repetition as second attribute is missing in %s" % (file_))

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            if not f_name in self.metainfo.feature_group_dict.keys():
                logger.error(
                    "Feature step \"%s\" was not defined in feature steps" % (
                        f_name))

        if len(self.metainfo.feature_group_dict.keys()) != len(
                arff_dict["attributes"][2:]):
            logger.error(
                "Number of feature steps in description.txt (%d) and feature_runstatus.arff (%d) does not match." % (
                    len(self.metainfo.feature_group_dict.keys()),
                    len(arff_dict["attributes"][2:-1])))

        pairs_inst_rep = []
        for data in arff_dict["data"]:
            inst_name = data[0]
            repetition = data[1]
            stati = data[2:]
            inst_ = self.instances.get(inst_name)
            if not inst_:
                logger.warning(
                    "Instance \"%s\" has feature step status but was not found in performance file" % (
                        inst_name))
                continue

            if (inst_name, repetition) in pairs_inst_rep:
                logger.warning("Pair (%s,%s) is not unique in %s" % (
                    inst_name, repetition, file_))
            else:
                pairs_inst_rep.append((inst_name, repetition))

            # ===================================================================
            # # if runstatus of feature vector is not always ok, remove feature vector
            # if reduce(lambda x,y: False if ((not x) and y.upper() == "OK") else True, stati, False):
            #     inst_._features = None
            #===================================================================
            for status, f_step in zip(stati, arff_dict["attributes"][2:]):
                inst_._features_status[f_step[0]] = status

                #inst_ = self.instances[inst_name] = self.instances.get(inst_name, Instance)

    def read_ground_truth(self, file_):
        '''
            read ground truths of all instances
            and save them in self.instances
            
            @RELATION GROUND_TRUTH_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE SATUNSAT {SAT,UNSAT}
            @ATTRIBUTE OPTIMAL_VALUE NUMERIC
        '''

        logger.info("Read %s" % (file_))

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "instance_id as first attribute is missing in %s" % (file_))

        # extract feature names
        for attr in arff_dict["attributes"][1:]:
            self.metainfo.ground_truths[attr[0]] = attr[1]

        insts = []
        for data in arff_dict["data"]:
            inst_name = data[0]
            truth = data[1:]

            inst_ = self.instances.get(inst_name)
            if not inst_:
                logger.warning(
                    "Instance \"%s\" has ground truths but was not found in performance file")
                continue

            truth_dict = dict((truth_name[0],
                               self.metainfo.ground_truths[truth_name[0]].index(
                                   truth_value) if truth_value else -1) \
                              for truth_name, truth_value in
                              zip(arff_dict["attributes"][1:], truth))
            inst_._ground_truth = truth_dict

            if inst_name in insts:
                logger.error(
                    "Instance \"%s\" is not unique in %s" % (inst_name, file_))
            else:
                insts.append(inst_name)

    def read_cv(self, file_):
        '''
            read cross validation <file_>
            
            @RELATION CV_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE fold NUMERIC
        '''
        logger.info("Read %s" % (file_))
        self.metainfo.cv_given = True

        fp = open(file_, "rb")
        arff_dict = arff.load(fp)
        fp.close()

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            logger.error(
                "repetition as second attribute is missing in %s" % (file_))
        if arff_dict["attributes"][2][0].upper() != "FOLD":
            logger.error(
                "fold as third attribute is missing in %s" % (file_))

        rep_fold_dict = {}
        for data in arff_dict["data"]:
            inst_name = data[0]
            rep = int(data[1])
            fold = int(data[2])

            inst_ = self.instances.get(inst_name)
            if not inst_:
                logger.warning(
                    "Instance \"%s\" has ground truths but was not found in performance file")
                continue

            inst_._fold[rep] = fold
            fold_distribution = rep_fold_dict.get(rep, {})
            rep_fold_dict[rep] = fold_distribution
            fold_distribution[fold] = fold_distribution.get(fold, 0)
            fold_distribution[fold] += 1

        for rep, fold_dist in rep_fold_dict.items():
            logger.info("%d-th repetition: %s distribution" % (
                rep, ",".join(map(str, list(fold_dist.values())))))


    def check_instances(self):
        '''
            check each instances of completeness and soundness
        '''
        logger.info("Check Consistency and Completeness of input data")

        n_instances = len(self.instances)
        n_no_feats = 0
        n_unsolvable = 0
        n_unsolvable2 = 0
        n_valid = 0
        n_presolved = 0
        feature_costs = 0

        for inst_ in self.instances.values():
            valid = True
            unsolvable = "OK" not in list(inst_._status.values())
            if unsolvable:
                n_unsolvable += 1
                valid = False
            if not inst_._cost:
                logger.error("Missing algorithm cost for instance \"%s\"" % (
                    inst_._name))
                valid = False
            inst_.finished_input(self.metainfo.algorithms)
            if not inst_._features:
                logger.info(
                    "Missing features values for instance \"%s\"" % (
                        inst_._name))
                n_no_feats += 1
                valid = False
            if inst_._pre_solved:
                n_presolved += 1
            if valid:
                n_valid += 1
            times = filter(lambda x: x < self.metainfo.algorithm_cutoff_time,
                           inst_._cost_vec)
            if not times:
                n_unsolvable2 += 1
            feature_costs += inst_._feature_cost_total

            if not times and not unsolvable:
                print(inst_._name)
                print(inst_._cost_vec)
                print(inst_._status.values())
                # ===================================================================
                # if not unsolvable and not times:
                #     print(inst_)
                #===================================================================
                #if not inst_._feature_runtimes:
                #    logger.error("Miss feature costs for instance %s" %(inst_._name))
                #if not inst_._stati:
                #    logger.error("Miss run status for instance %s" %(inst_._name), type_="w")
                #if not inst_._ground_truth:
                #    logger.error("Miss ground truth for instance %s" %(inst_._name), type_="w")

        logger.info("Instances: \t\t %d" % (n_instances))
        logger.info("Algorithms: \t\t %d" % (len(self.metainfo.algorithms)))
        logger.info(
            "Incomplete Feature Vector: \t %d (maybe imputed)" % (n_no_feats))
        logger.info("Unsolvable Instances (status): \t %d" % (n_unsolvable))
        logger.info(
            "Unsolvable Instances (runtime): \t %d" % (n_unsolvable2))
        logger.info("Valid Instances: \t %d" % (n_valid))
        logger.info("Presolved: \t\t %d" % (n_presolved))
        logger.info(
            "Average Feature Costs: \t %.4f" % (feature_costs / n_instances))

        if n_unsolvable != n_unsolvable2:
            logger.warning(
                "Number of unsolvable instances regarding status and runtime is not consistent.")

        if not n_valid:
            logger.error("Have not found valid instances")

    def check_feature_status(self):
        '''
            check that features are Na
        '''
        feature_group_dict = self.metainfo.feature_group_dict

        for inst_ in self.instances.values():
            not_ok_steps = []
            for step, status in inst_._features_status.items():
                if status.upper() != "OK":
                    not_ok_steps.append(step)

            unused_features = set()
            for u_step in not_ok_steps:
                not_processed_features = feature_group_dict[u_step]
                unused_features = unused_features.union(
                    set(not_processed_features))
            not_ok_index_features = sorted(list(
                map(str, self.metainfo.features).index(un_feature) for
                un_feature in unused_features), reverse=True)
            ok_index_features = set(
                range(len(self.metainfo.features))).difference(
                not_ok_index_features)

            warned = False
            for indx in not_ok_index_features:
                if inst_._features[indx] is not None:
                    if not warned:
                        logger.warning(
                            "Not all features of %s are NA although the corresponding feature step is not OK." % (
                                inst_._name))
                        warned = True
                        # inst_._features[indx] = None

            ok_values = [inst_._features[indx] for indx in ok_index_features]
            if None in ok_values:
                logger.error(
                    "Missing Features with status OK: %s." % (inst_._name))

    def remove_features(self):
        '''
            inst_dict: instance name -> Instance()
            meta_info: parsed coseal meta information and command line arguments (meta_info.options)
        '''

        feature_steps = self.metainfo.options.feature_steps
        feature_group_dict = self.metainfo.feature_group_dict

        if not feature_steps:
            feature_steps = list(
                self.metainfo.feature_steps)  # if no steps are specified, use default

        logger.info("Used Feature Steps: %s" % (",".join(feature_steps)))
        empty_check = set(feature_steps).difference(
            set(feature_group_dict.keys()))
        if empty_check:
            logger.error(
                "Feature steps (--feature-steps [list]) are not defined in data: %s" % (
                    ",".join(empty_check)), -2)

        unused_features = set()
        unused_steps = set(feature_group_dict.keys()).difference(
            set(feature_steps))
        for u_step in unused_steps:
            not_processed_features = feature_group_dict[u_step]
            unused_features = unused_features.union(set(not_processed_features))

        logger.info(
            "Remove features: %s\n" % (",".join(unused_features)))
        used_features = set(self.metainfo.features).difference(unused_features)
        logger.info("Used features: %s\n" % (",".join(used_features)))

        unused_index_features = sorted(list(
            map(str, self.metainfo.features).index(un_feature) for un_feature in
            unused_features), reverse=True)

        for inst_ in self.instances.values():
            for un_feature_indx in unused_index_features:
                inst_._features.pop(un_feature_indx)
            total_cost = 0
            previous_presolved = False
            for f_step in feature_steps:
                if inst_._feature_group_cost_dict.get(
                        f_step) and not previous_presolved:  # feature costs are maybe None
                    total_cost += inst_._feature_group_cost_dict[f_step]
                if inst_._features_status[f_step] == "PRESOLVED":
                    previous_presolved = True
            for un_step in unused_steps:  # remove step status if unused
                del inst_._features_status[un_step]

            inst_._feature_cost_total = total_cost
            inst_._pre_solved = "PRESOLVED" in map(lambda x: x.upper(),
                                                   inst_._features_status.values())

        for un_feature_indx in unused_index_features:
            self.metainfo.features.pop(un_feature_indx)

        if self.metainfo.options.impute == "none":
            for inst_ in self.instances.values():
                if reduce(lambda x, y: False if (
                            (not x) and y.upper() == "OK") else True,
                          inst_._features_status.values(), False):
                    inst_._features = None

                    # ===============================================================================
                    # if __name__ == '__main__':
                    #
                    #     checker = CosealReader()
                    #     checker.main()
                    #===============================================================================