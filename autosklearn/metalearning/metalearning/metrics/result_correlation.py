from collections import OrderedDict
import itertools

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.utils

import pyMetaLearn.metalearning.create_datasets as create_datasets
from pyMetaLearn.metalearning.meta_base import Run


def get_result_correlation_metric(metafeatures, runs,
                                  n_estimators=100, max_features='auto',
                                  min_samples_split=2, min_samples_leaf=1,
                                  n_jobs=1, random_state=None, oob_score=False):
    model = LearnedDistanceRF(n_estimators=n_estimators,
                              max_features=max_features,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              n_jobs=n_jobs, random_state=random_state,
                              oob_score=oob_score)
    model.fit(metafeatures, runs)

    def distance_function(d1, d2):
        x = np.hstack((d1, d2))
        predictions = model.predict(x)
        predictions = model.predict(x)

        # Predictions are between -1 and 1, -1 indicating a negative correlation.
        # Since we evaluate the dataset with the smallest metric, we would
        # evaluate the dataset with the most negative correlation
        # logger.info(predictions)
        # logger.info(predictions[0] * -1)
        return (predictions[0] * -1) + 1

    return distance_function


class LearnedDistanceRF(object):
    # TODO: instead of a random forest, the user could provide a generic
    # import call with which it is possible to import a class which
    # implements the sklearn fit and predict function...
    def __init__(self, n_estimators=100, max_features=0.2,
                 min_samples_split=2, min_samples_leaf=1, n_jobs=1,
                 random_state=None, oob_score=False):
        if isinstance(random_state, str):
            random_state = int(random_state)
        rs = sklearn.utils.check_random_state(random_state)
        rf = sklearn.ensemble.RandomForestRegressor(
            n_estimators=int(n_estimators), max_features=float(max_features),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            criterion="mse", random_state=rs, oob_score=oob_score,
            n_jobs=int(n_jobs))
        self.model = rf

    def fit(self, metafeatures, runs):
        X, Y = self._create_dataset(metafeatures, runs)
        model = self._fit(X, Y)
        return model

    def _create_dataset(self, metafeatures, runs):
        runs = self._apply_surrogates(metafeatures, runs)
        X, Y = create_datasets.create_predict_spearman_rank(
            metafeatures, runs, "permutation")
        return X, Y

    def _fit(self, X, Y):
        self.model.fit(X, Y)
        return self.model

    def _apply_surrogates(self, metafeatures, runs, n_estimators=500,
                          n_jobs=1, random_state=None, oob_score=True):

        # Find out all configurations for which we need to know result in
        # order to calculate the correlation coefficient
        configurations = set()
        configurations_per_run = dict()
        outcomes_per_run = dict()
        for name in runs:
            configurations_per_run[name] = set()
            outcomes_per_run[name] = dict()
            for experiment in runs[name]:
                # TODO: refactor the classes so params are hashable
                configurations.add(str(experiment.configuration))
                configurations_per_run[name].add(str(experiment.configuration))
                outcomes_per_run[name][str(experiment.configuration)] = \
                    experiment.result

        filled_runs = {}
        for name in runs:
            print(".",)
            run = runs[name]
            # Transfer all previous experiments
            filled_runs[name] = run

            train_x = []
            train_y = []
            predict = []

            for configuration in configurations:
                param = eval(configuration)
                if configuration in configurations_per_run[name]:
                    train_x.append(param)
                    train_y.append(outcomes_per_run[name][configuration])
                else:
                    predict.append(param)

            train_x = pd.DataFrame(train_x)
            train_y = pd.Series(train_y)
            predict = pd.DataFrame(predict)

            # Hacky procedure to be able to use the scaling/onehotencoding on
            # all data at the same time
            stacked = train_x.append(predict, ignore_index=True)
            stacked_y = train_y.append(pd.Series(np.zeros((len(predict)))))

            if len(predict) == 0:
                continue

            if isinstance(random_state, str):
                random_state = int(random_state)
            rs = sklearn.utils.check_random_state(random_state)
            rf = sklearn.ensemble.RandomForestRegressor(
                n_estimators=int(n_estimators),
                criterion="mse", random_state=rs, oob_score=oob_score,
                n_jobs=int(n_jobs))

            # For the y array we have to convert the NaNs already here; maybe
            #  it would even better to leave them out...
            stacked_y.fillna(1, inplace=True)

            stacked, stacked_y = _convert_pandas_to_npy(stacked, stacked_y)
            num_training_samples = len(train_x)
            train_x = stacked[:num_training_samples]
            predict_x = stacked[num_training_samples:]
            train_y = stacked_y[:num_training_samples]

            rf = rf.fit(train_x, train_y)

            prediction = rf.predict(predict_x)
            for x_, y_ in itertools.izip(predict.iterrows(), prediction):
                # Remove all values which are nan
                params = {pair[0]: pair[1] for pair in x_[1].to_dict().items()
                          if pair[1] == pair[1]}
                params = OrderedDict([pair for pair in sorted(params.items())])
                # TODO: add a time prediction surrogate
                filled_runs[name].append(Run(params, y_, 1))
            filled_runs[name].sort()

        return filled_runs

    def predict(self, metafeatures):
        assert isinstance(metafeatures, np.ndarray)
        return self.model.predict(metafeatures)
