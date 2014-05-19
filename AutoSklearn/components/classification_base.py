class AutoSklearnClassificationAlgorithm(object):
    def __init__(self):
        self.estimator = None

    def handles_missing_values(self):
        raise NotImplementedError()

    def handles_nominal_features(self):
        raise NotImplementedError()

    def handles_numeric_features(self):
        raise NotImplementedError()

    def handles_non_binary_classes(self):
        raise NotImplementedError()

    def get_hyperparameter_search_space(self):
        raise NotImplementedError()

    def fit(self, X, Y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def get_estimator(self):
        return self.estimator

    def __str__(self):
        raise NotImplementedError()
