import numpy as np
import sklearn.ensemble


class OneVSOneSelector:
    def __init__(self, configuration, default_strategy_idx, rng):
        self.configuration = configuration
        self.default_strategy_idx = default_strategy_idx
        self.rng = rng
        self.models = None
        self.target_indices = None
        self.selectors_ = None
        self.weights_ = {}
        self.X_train = None

    def fit(self, X, y, methods, minima, maxima):
        self.X_train = X.copy()
        target_indices = np.array(list(range(y.shape[1])))
        models = dict()
        weights = dict()
        for i in range(len(target_indices)):
            models[i] = dict()
            weights[i] = dict()
            for j in range(i + 1, len(target_indices)):
                y_i_j = y[:, i] < y[:, j]
                min_i = np.array([minima[methods[i]][task_id] for task_id in X.index])
                max_i = np.array([maxima[methods[i]][task_id] for task_id in X.index])
                min_j = np.array([minima[methods[j]][task_id] for task_id in X.index])
                max_j = np.array([maxima[methods[j]][task_id] for task_id in X.index])

                minimum = np.minimum(min_i, min_j)
                maximum = np.maximum(max_i, max_j)
                diff = maximum - minimum
                diff[diff == 0] = 1
                normalized_y_i = (y[:, i].copy() - minimum) / diff
                normalized_y_j = (y[:, j].copy() - minimum) / diff

                weights_i_j = np.abs(normalized_y_i - normalized_y_j)
                if np.all([target == y_i_j[0] for target in y_i_j]):
                    n_zeros = int(np.ceil(len(y_i_j) / 2))
                    n_ones = int(np.floor(len(y_i_j) / 2))
                    base_model = sklearn.dummy.DummyClassifier(
                        strategy='constant', constant=y_i_j[0],
                    )
                    base_model.fit(
                        X.values,
                        np.array(([[0]] * n_zeros) + ([[1]] * n_ones)).flatten(),
                        sample_weight=weights_i_j,
                    )
                else:
                    base_model = sklearn.ensemble.RandomForestClassifier(
                        random_state=self.rng,
                        n_estimators=500,
                        oob_score=True,
                        bootstrap=True,
                        min_samples_split=self.configuration['min_samples_split'],
                        min_samples_leaf=self.configuration['min_samples_leaf'],
                        max_features=int(np.rint(X.shape[1] ** self.configuration['max_features'])),
                    )
                    base_model.fit(X.values, y_i_j, sample_weight=weights_i_j)
                models[i][j] = base_model
                weights[i][j] = weights_i_j
        self.models = models
        self.weights_ = weights
        self.target_indices = target_indices

    def predict(self, X):

        if self.default_strategy_idx is not None:
            use_prediction = False
            counter = 0
            te = X.copy().flatten()
            assert len(te) == 3
            for _, tr in self.X_train.iterrows():
                tr = tr.to_numpy()
                if tr[0] >= te[0] and tr[1] >= te[1] and tr[2] >= te[2]:
                    counter += 1

            if counter > 0:
                use_prediction = True

            if not use_prediction:
                print('Using Backup selector')
                return np.array(
                    [1 if i == self.default_strategy_idx else 0 for i in self.target_indices]
                )
            print('Using no backup selector')

        X = X.reshape((1, -1))

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                raw_predictions[(i, j)] = self.models[i][j].predict(X)

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions

    def predict_oob(self, X):

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                rp = self.models[i][j].oob_decision_function_.copy()
                rp[np.isnan(rp)] = 0
                rp = np.nanargmax(rp, axis=1)
                raw_predictions[(i, j)] = rp

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions
