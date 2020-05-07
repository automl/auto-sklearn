from ConfigSpace.configuration_space import ConfigurationSpace


class Rescaling(object):
    # Rescaling does not support fit_transform (as of 0.19.1)!

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
