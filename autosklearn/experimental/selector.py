import typing

import copy
import itertools

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.ensemble


class AbstractSelector:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        minima: typing.Dict[int, typing.Dict[str, float]],
        maxima: typing.Dict[int, typing.Dict[str, float]],
    ) -> None:
        raise NotImplementedError()

    def predict(
        self, X: pd.DataFrame, y: typing.Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        prediction = self._predict(X, y)
        for col, series in prediction.iteritems():
            assert series.dtype == float, (col, series)
        np.testing.assert_array_almost_equal(
            prediction.sum(axis="columns").to_numpy(),
            np.ones(X.shape[0]),
            err_msg=prediction.to_csv(),
        )
        return prediction

    def _predict(
        self, X: pd.DataFrame, y: typing.Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        raise NotImplementedError()


class OneVSOneSelector(AbstractSelector):
    def __init__(self, configuration, random_state, tie_break_order):
        self.configuration = configuration
        self.single_strategy_idx = None
        self.rng = random_state
        self.tie_break_order = tie_break_order
        self.models = None
        self.target_indices = None
        self.selectors_ = None
        self.weights_ = {}
        self.strategies_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        minima: typing.Dict[int, typing.Dict[str, float]],
        maxima: typing.Dict[int, typing.Dict[str, float]],
    ) -> None:

        self.maxima = copy.deepcopy(maxima)
        target_indices = np.array(list(range(y.shape[1])))
        models = dict()
        weights = dict()

        self.strategies_ = copy.deepcopy(y.columns.to_list())
        y_pd = y.copy()
        y = y.to_numpy()

        for i in range(len(target_indices)):
            models[i] = dict()
            weights[i] = dict()
            for j in range(i + 1, len(target_indices)):

                if self.configuration["normalization"] in (
                    "all",
                    "binary",
                    "y",
                    "all1",
                    "binary1",
                ):
                    minimum2 = np.ones(len(X)) * np.inf
                    maximum2 = np.zeros(len(X))

                    if self.configuration["normalization"] in ("all", "all1"):
                        for idx, task_id in enumerate(X.index):
                            for method_id in range(len(target_indices)):
                                minimum2[idx] = np.nanmin(
                                    (
                                        minimum2[idx],
                                        minima[task_id][self.strategies_[method_id]],
                                    )
                                )
                                maximum2[idx] = np.nanmax(
                                    (
                                        maximum2[idx],
                                        maxima[task_id][self.strategies_[method_id]],
                                    )
                                )
                        if self.configuration["normalization"] == "all1":
                            maximum2 = np.ones_like(maximum2)
                    elif self.configuration["normalization"] in ("binary", "binary1"):
                        for idx, task_id in enumerate(X.index):
                            for method_id in (i, j):
                                minimum2[idx] = np.nanmin(
                                    (
                                        minimum2[idx],
                                        minima[task_id][self.strategies_[method_id]],
                                    )
                                )
                                maximum2[idx] = np.nanmax(
                                    (
                                        maximum2[idx],
                                        maxima[task_id][self.strategies_[method_id]],
                                    )
                                )
                        if self.configuration["normalization"] == "binary1":
                            maximum2 = np.ones_like(maximum2)
                    elif self.configuration["normalization"] == "y":
                        for idx, task_id in enumerate(X.index):
                            minimum2[idx] = np.nanmin(
                                (minimum2[idx], y_pd.loc[task_id].min())
                            )
                            maximum2[idx] = np.nanmax(
                                (maximum2[idx], y_pd.loc[task_id].max())
                            )
                    else:
                        raise ValueError(self.configuration["normalization"])

                    y_i_j = y[:, i] < y[:, j]
                    mask = np.isfinite(y[:, i]) & np.isfinite(y[:, j])
                    X_ = X.to_numpy()[mask]
                    y_i_j = y_i_j[mask]

                    minimum = minimum2[mask]
                    maximum = maximum2[mask]

                    diff = maximum - minimum
                    diff[diff == 0] = 1
                    normalized_y_i = (y[:, i][mask].copy() - minimum) / diff
                    normalized_y_j = (y[:, j][mask].copy() - minimum) / diff

                    weights_i_j = np.abs(normalized_y_i - normalized_y_j)

                elif self.configuration["normalization"] == "rank":
                    y_i_j = y[:, i] < y[:, j]
                    mask = np.isfinite(y[:, i]) & np.isfinite(y[:, j])
                    X_ = X.to_numpy()[mask]
                    y_i_j = y_i_j[mask]
                    ranks = scipy.stats.rankdata(y[mask], axis=1)
                    weights_i_j = np.abs(ranks[:, i] - ranks[:, j])

                elif self.configuration["normalization"] == "None":
                    y_i_j = y[:, i] < y[:, j]
                    mask = np.isfinite(y[:, i]) & np.isfinite(y[:, j])
                    X_ = X.to_numpy()[mask]
                    y_i_j = y_i_j[mask]
                    weights_i_j = np.ones_like(y_i_j).astype(int)

                else:
                    raise ValueError(self.configuration["normalization"])

                if len(y_i_j) == 0:
                    models[i][j] = None
                    weights[i][j] = None
                    continue

                if np.all([target == y_i_j[0] for target in y_i_j]):
                    n_zeros = int(np.ceil(len(y_i_j) / 2))
                    n_ones = int(np.floor(len(y_i_j) / 2))
                    import sklearn.dummy

                    base_model = sklearn.dummy.DummyClassifier(
                        strategy="constant", constant=y_i_j[0]
                    )
                    base_model.fit(
                        X_,
                        np.array(([[0]] * n_zeros) + ([[1]] * n_ones)).flatten(),
                        sample_weight=weights_i_j,
                    )
                else:
                    if self.configuration.get("max_depth") == 0:
                        import sklearn.dummy

                        loss_i = np.sum((y_i_j == 0) * weights_i_j)
                        loss_j = np.sum((y_i_j == 1) * weights_i_j)

                        base_model = sklearn.dummy.DummyClassifier(
                            strategy="constant",
                            constant=1 if loss_i < loss_j else 0,
                        )
                        base_model.fit(
                            X_,
                            np.ones_like(y_i_j) * int(loss_i < loss_j),
                            sample_weight=weights_i_j,
                        )
                    else:
                        base_model = self.fit_pairwise_model(
                            X_,
                            y_i_j,
                            weights_i_j,
                            self.rng,
                            self.configuration,
                        )
                models[i][j] = base_model
                weights[i][j] = weights_i_j
        self.models = models
        self.weights_ = weights
        self.target_indices = target_indices

    def _predict(
        self, X: pd.DataFrame, y: typing.Optional[pd.DataFrame]
    ) -> pd.DataFrame:

        if y is not None:
            raise ValueError("y must not be provided")

        raw_predictions = dict()
        raw_probas = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                if self.models[i][j] is not None:
                    raw_predictions[(i, j)] = self.models[i][j].predict(X)
                    raw_probas[(i, j)] = self.models[i][j].predict_proba(X)

        if len(raw_predictions) == 0:
            predictions = pd.DataFrame(
                0, index=X.index, columns=self.strategies_
            ).astype(float)
            predictions.iloc[:, self.single_strategy_idx] = 1.0
            return predictions

        predictions = {}
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    if (i, j) in raw_predictions:
                        if self.configuration["prediction"] == "soft":
                            if raw_probas[(i, j)].shape[1] == 1:
                                proba = raw_probas[(i, j)][x_idx][0]
                            else:
                                proba = raw_probas[(i, j)][x_idx][1]
                            wins[i] += proba
                            wins[j] += 1 - proba
                        elif self.configuration["prediction"] == "hard":
                            prediction = raw_predictions[(i, j)][x_idx]
                            if prediction == 1:
                                wins[i] += 1
                            else:
                                wins[j] += 1
                        else:
                            raise ValueError(self.configuration["prediction"])

            n_prev = np.inf
            # Tie breaking
            while True:
                most_wins = np.max(wins)
                most_wins_mask = most_wins == wins
                if n_prev == np.sum(most_wins_mask):
                    n_prev = np.sum(most_wins_mask)
                    hit = False
                    for method in self.tie_break_order:
                        if method not in self.strategies_:
                            continue
                        method_idx = self.strategies_.index(method)
                        if most_wins_mask[method_idx]:
                            wins[method_idx] += 1
                            hit = True
                            break
                    if not hit:
                        wins[
                            int(self.rng.choice(np.argwhere(most_wins_mask).flatten()))
                        ] += 1
                elif np.sum(most_wins_mask) > 1:
                    n_prev = np.sum(most_wins_mask)
                    where = np.argwhere(most_wins_mask).flatten()
                    for i, j in itertools.combinations(where, 2):
                        if (i, j) in raw_predictions:
                            prediction = raw_predictions[(i, j)][x_idx]
                            if prediction == 1:
                                wins[i] += 1
                            else:
                                wins[j] += 1
                        else:
                            method_i = self.strategies_[i]
                            method_j = self.strategies_[j]
                            if self.tie_break_order.index(
                                method_i
                            ) < self.tie_break_order.index(method_j):
                                wins[i] += 1
                            else:
                                wins[j] += 1
                else:
                    break

            wins = wins / np.sum(wins)
            predictions[X.index[x_idx]] = wins

        return_value = {
            task_id: {
                strategy: predictions[task_id][strategy_idx]
                for strategy_idx, strategy in enumerate(self.strategies_)
            }
            for task_id in X.index
        }
        return_value = pd.DataFrame(return_value).transpose().astype(float)
        return_value = return_value[self.strategies_]
        return_value = return_value.fillna(0.0)
        return return_value

    def fit_pairwise_model(self, X, y, weights, rng, configuration):
        raise NotImplementedError()


class OVORF(OneVSOneSelector):
    def __init__(self, n_estimators, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators

    def fit_pairwise_model(self, X, y, weights, rng, configuration):
        base_model = sklearn.ensemble.RandomForestClassifier(
            random_state=rng,
            n_estimators=self.n_estimators,
            bootstrap=True if configuration["bootstrap"] == "True" else False,
            min_samples_split=configuration["min_samples_split"],
            min_samples_leaf=configuration["min_samples_leaf"],
            max_features=int(configuration["max_features"]),
            max_depth=configuration["max_depth"],
        )
        base_model.fit(X, y, sample_weight=weights)
        return base_model


class FallbackWrapper(AbstractSelector):
    def __init__(self, selector, default_strategies: typing.List[str]):
        self.selector = selector
        self.default_strategies = default_strategies

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        minima: typing.Dict[int, typing.Dict[str, float]],
        maxima: typing.Dict[int, typing.Dict[str, float]],
    ) -> None:
        self.X_ = X
        self.strategies_ = y.columns
        self.return_value_ = np.array(
            [
                (len(self.strategies_) - self.default_strategies.index(strategy) - 1)
                / (len(self.strategies_) - 1)
                for strategy in self.strategies_
            ]
        )
        self.return_value_ = self.return_value_ / np.sum(self.return_value_)
        self.selector.fit(X, y, minima, maxima)

    def _predict(
        self, X: pd.DataFrame, y: typing.Optional[pd.DataFrame]
    ) -> pd.DataFrame:

        if y is not None:
            prediction = self.selector.predict(X, y)
        else:
            prediction = self.selector.predict(X)
        for task_id, x in X.iterrows():
            counter = 0
            te = x.copy()
            assert len(te) == self.X_.shape[1]
            for _, tr in self.X_.iterrows():
                tr = tr.to_numpy()
                if all([tr[i] >= te[i] for i in range(self.X_.shape[1])]):
                    counter += 1

            if counter == 0:
                prediction.loc[task_id] = pd.Series(
                    {
                        strategy: value
                        for strategy, value in zip(self.strategies_, self.return_value_)
                    }
                )

        return prediction
