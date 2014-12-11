class AutoSklearnClassificationAlgorithm(object):
    """Provide an abstract interface for classification algorithms in
    AutoSklearn.

    Make a subclass of this and put it into the directory
    `AutoSklearn/components/classification` to make it available."""
    def __init__(self):
        self.estimator = None
        self.properties = None

    def get_properties(self):
        """Get the properties of the underlying algorithm. These are:

        * Can the algorithm handle missing values?
          (handles_missing_values : {True, False})
        * Can the algorithm handle nominal features?
          (handles_nominal_features : {True, False})
        * Can the algorithm handle numerical features?
          (handles_numerical_features : {True, False})
        * Can the algorithm handle multiclass-classification problems?
          (handles_multiclass : {True, False})
        * Can the algorithm handle multilabel-classification problems?
          (handles_multilabel : {True, False}
        * Is the algorithm deterministic for a given seed?
          (is_deterministic : {True, False)
        * Can the algorithm handle sparse data?
          (handles_sparse : {True, False}
        * What are the preferred types of the data array?
          (preferred_dtype : list of tuples)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    def get_hyperparameter_search_space(self):
        """Return the configuration space of this classification algorithm.

        Returns
        -------
        HPOlibConfigspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = [n_samples]

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator

    def __str__(self):
        raise NotImplementedError()
