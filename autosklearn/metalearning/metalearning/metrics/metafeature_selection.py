def _mfs_l1(self, d1, d2):
    d1 = d1.copy() * self.model.weights
    d2 = d2.copy() * self.model.weights
    return self._l1(d1, d2)


def _mfw_l1(self, d1, d2):
    return self._mfs_l1(d1, d2)


"""
elif self.metric == 'mfs_l1':
    # This implements metafeature selection as described by Matthias
    # Reif in 'Metalearning for evolutionary parameter optimization
    # of classifiers'
    self.model = MetaFeatureSelection(**self.metric_kwargs)
    return self.model.fit(metafeatures, runs)
elif self.metric == 'mfw_l1':
    self.model = MetaFeatureSelection(mode='weight', **self.metric_kwargs)
    return self.model.fit(metafeatures, runs)
"""

"""
class MetaFeatureSelection(object):
    def __init__(self, max_number_of_combinations=10, random_state=None,
                 k=1, max_features=0.5, mode='select'):
        self.max_number_of_combinations = max_number_of_combinations
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.k = k
        self.max_features = max_features
        self.weights = None
        self.mode = mode

    def fit(self, metafeatures, runs):
        self.datasets = metafeatures.index
        self.all_other_datasets = {}  # For faster indexing
        self.all_other_runs = {}  # For faster indexing
        self.parameter_distances = defaultdict(dict)
        self.best_configuration_per_dataset = {}
        self.mf_names = metafeatures.columns
        self.kND = KNearestDatasets(metric='l1')

        for dataset in self.datasets:
            self.all_other_datasets[dataset] = \
                pd.Index([name for name in self.datasets if name != dataset])

        for dataset in self.datasets:
            self.all_other_runs[dataset] = \
                {key: runs[key] for key in runs if key != dataset}

        for dataset in self.datasets:
            self.best_configuration_per_dataset[dataset] = \
                sorted(runs[dataset], key=lambda t: t.result)[0]

        for d1, d2 in itertools.combinations(self.datasets, 2):
            hps1 = self.best_configuration_per_dataset[d1]
            hps2 = self.best_configuration_per_dataset[d2]
            keys = set(hps1.params.keys())
            keys.update(hps2.params.keys())
            dist = 0
            for key in keys:
                # TODO: test this; it can happen that string etc occur
                try:
                    p1 = float(hps1.params.get_value(key, 0))
                    p2 = float(hps2.params.get_value(key, 0))
                    dist += abs(p1 - p2)
                except:
                    dist += 0 if hps1.params.get_value(key, 0) == \
                                 hps2.params.get_value(key, 0) else 1

                    # dist += abs(hps1.params.get_value(key, 0) - hps2.params.get_value(key, 0))
            self.parameter_distances[d1][d2] = dist
            self.parameter_distances[d2][d1] = dist

        if self.mode == 'select':
            self.weights = self._fit_binary_weights(metafeatures)
        elif self.mode == 'weight':
            self.weights = self._fit_weights(metafeatures)

        sys.stderr.write(str(self.weights))
        sys.stderr.write('\n')
        sys.stderr.flush()
        return self.weights

    def _fit_binary_weights(self, metafeatures):
        best_selection = None
        best_distance = sys.maxint

        for i in range(2,
                       int(np.round(len(self.mf_names) * self.max_features))):
            sys.stderr.write(str(i))
            sys.stderr.write('\n')
            sys.stderr.flush()

            combinations = []
            for j in range(self.max_number_of_combinations):
                combination = []
                target = i
                maximum = len(self.mf_names)
                while len(combination) < target:
                    random = self.random_state.randint(maximum)
                    name = self.mf_names[random]
                    if name not in combination:
                        combination.append(name)

                combinations.append(pd.Index(combination))

            for j, combination in enumerate(combinations):
                dist = 0
                for dataset in self.datasets:
                    hps = self.best_configuration_per_dataset[dataset]
                    self.kND.fit(metafeatures.loc[self.all_other_datasets[
                                                      dataset], combination],
                                 self.all_other_runs[dataset])
                    nearest_datasets = self.kND.kBestSuggestions(
                        metafeatures.loc[dataset, np.array(combination)],
                        self.k)
                    for nd in nearest_datasets:
                        # print "HPS", hps.params, "nd", nd[2]
                        dist += self.parameter_distances[dataset][nd[0]]

                if dist < best_distance:
                    best_distance = dist
                    best_selection = combination

        weights = dict()
        for metafeature in metafeatures:
            if metafeature in best_selection:
                weights[metafeature] = 1
            else:
                weights[metafeature] = 0
        return pd.Series(weights)

    def _fit_weights(self, metafeatures):
        best_weights = None
        best_distance = sys.maxint

        def objective(weights):
            dist = 0
            for dataset in self.datasets:
                self.kND.fit(metafeatures.loc[self.all_other_datasets[
                    dataset], :] * weights, self.all_other_runs[dataset])
                nearest_datasets = self.kND.kBestSuggestions(
                    metafeatures.loc[dataset, :] * weights, self.k)
                for nd in nearest_datasets:
                    dist += self.parameter_distances[dataset][nd[0]]

            return dist

        for i in range(10):
            w0 = np.ones((len(self.mf_names, ))) * 0.5 + \
                 (np.random.random(size=len(self.mf_names)) - 0.5) * i / 10
            bounds = [(0, 1) for idx in range(len(self.mf_names))]

            res = scipy.optimize.minimize \
                (objective, w0, bounds=bounds, method='L-BFGS-B',
                 options={'disp': True})

            if res.fun < best_distance:
                best_distance = res.fun
                best_weights = pd.Series(res.x, index=self.mf_names)

        return best_weights
"""
