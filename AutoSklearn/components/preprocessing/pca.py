import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from ..preprocessor_base import AutoSklearnPreprocessingAlgorithm


class PCA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, keep_variance, whiten, random_state=None):
        self.keep_variance = keep_variance
        self.whiten = whiten

    def fit(self, X, Y):
        # TODO: implement that keep_variance can be a percentage (in int)
        self.preprocessor = sklearn.decomposition.PCA(whiten=self.whiten,
                                                      copy=True)
                                                      # num components is
                                                      # selected further down
                                                      #  the code
        self.preprocessor.fit(X, Y)

        sum_ = 0.
        idx = 0
        while idx < len(self.preprocessor.explained_variance_ratio_) and \
                sum_ < self.keep_variance:
            sum_ += self.preprocessor.explained_variance_ratio_[idx]
            idx += 1

        components = self.preprocessor.components_
        self.preprocessor.components_ = components[:idx]
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    def handles_missing_values(self):
        return False

    def handles_nominal_features(self):
        return False

    def handles_numeric_features(self):
        return True

    def handles_non_binary_classes(self):
        return True

    @staticmethod
    def get_hyperparameter_search_space():
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 1.0, default=1.0)
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(keep_variance)
        cs.add_hyperparameter(whiten)
        return cs

    @staticmethod
    def get_all_accepted_hyperparameter_names():
        return (["keep_variance", "whiten"])

    def __str__(self):
        return "AutoSklearn Principle Component Analysis preprocessor."
