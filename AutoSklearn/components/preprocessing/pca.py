import sklearn.decomposition

from ...util import hp_uniform, hp_choice
from ..preprocessor_base import AutoSklearnPreprocessingAlgorithm

class PCA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, keep_variance=1.0, whiten=False, random_state=None):
        self.keep_variance = keep_variance
        self.whiten = whiten

    def fit(self, X, Y):
        self.preprocessor = sklearn.decomposition.PCA(whiten=self.whiten,
                                                      copy=True)
        self.preprocessor.fit(X, Y)

        sum_ = 0.
        idx = 0
        while idx < len(self.preprocessor.explained_variance_ratio_) and \
                sum_ < self.keep_variance:
            sum_ += self.preprocessor.explained_variance_ratio_[idx]
            idx += 1

        components = self.preprocessor.components_
        self.preprocessor.components_ = components[:idx]

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
        keep_variance = hp_uniform("n_components", 0.5, 1.0)
        whiten = hp_choice("whiten", ["False", "True"])
        return {"name": "pca", "keep_variance": keep_variance,
                "whiten": whiten}

    def __str__(self):
        return "AutoSklearn Principle Component Analysis preprocessor."