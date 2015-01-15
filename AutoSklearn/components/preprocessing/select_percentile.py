import sklearn.feature_selection


class SelectPercentileBase(object):

    def fit(self, X, Y):
        self.preprocessor = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func,
            percentile=self.percentile)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)
