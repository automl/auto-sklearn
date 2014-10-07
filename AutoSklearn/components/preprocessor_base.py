class AutoSklearnPreprocessingAlgorithm(object):
    def __init__(self):
        self.estimator = None

    def handles_missing_values(self):
        """Can the underlying algorithm handle missing values itself?

        Returns
        -------
        flag : Boolean
            True if the underlying algorithm handles missing values itself,
            otherwise False.

        Note
        ----

        This feature is not implemented yet. Missing values are not supported.
        """
        raise NotImplementedError()

    def handles_nominal_features(self):
        """Can the underlying algorithm handle nominal features?

        Returns
        -------
        flag : Boolean
            True if the underlying algorithm handles nominal values itself,
            otherwise False.

        Note
        ----

        This feature is not implemented yet. Nominal values are not
        supported. It is suggested to perform a OneHotEncoding on them.
        """
        raise NotImplementedError()

    def handles_numeric_features(self):
        """Can the underlying algorithm handle numeric features itself?

        Returns
        -------
        flag : Boolean
            True if the underlying algorithm handles numeric features itself,
            otherwise False.

        Note
        ----

        This feature is not implemented yet. Every algorithm support numeric
        features.
        """
        raise NotImplementedError()

    def get_hyperparameter_search_space(self):
        """Return the configuration space of this preprocessing algorithm.

        Returns
        -------
        cs : dict
            A dictionary with all hyperparameters as hyperopt.pyll objects.

        """
        raise NotImplementedError()

    def get_all_accepted_hyperparameter_names(self):
        """Return the name of all hyperparameters accepted by this preprocessing
         algorithm.

        This must not be the same as the list returned by
        :meth:`get_hyperparameter_search_space`. An example can be found in
        the components for the linear svm and the libsvm, where it is also
        possible to specifiy the parameters as the exponent to the base two.

        This list is used by the
        :class:`AutoSklearn.autosklearn.AutoSklearnClassifier` to check if it
        is called with illegal hyperparameters.

        Returns
        -------
        names : A list of accepted hyperparameter names.
        """
        raise NotImplementedError()

    def fit(self, X, Y):
        """The fit function calls the fit function of the underlying
        scikit-learn preprocessing algorithm and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = [n_samples]

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def transform(self, X):
        """The predict function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_preprocessor(self):
        """Return the underlying preprocessor object.

        Returns
        -------
        preprocessor : the underlying preprocessor object
        """
        return self.estimator

    def __str__(self):
        raise NotImplementedError()
