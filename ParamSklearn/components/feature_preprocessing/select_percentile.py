class SelectPercentileBase(object):

    def fit(self, X, y):
        import sklearn.feature_selection

        self.preprocessor = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func,
            percentile=self.percentile)

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        Xt = self.preprocessor.transform(X)
        if Xt.shape[1] == 0:
            raise ValueError("%s removed all features." % self.__class__.__name__)
        return Xt
