class AutoSklearnPreprocessingAlgorithm(object):
    """Provide an abstract interface for preprocessing algorithms in
    AutoSklearn.

    Make a subclass of this and put it into the directory
    `AutoSklearn/components/preprocessing` to make it available."""
    def __init__(self):
        self.preprocessor = None

    def get_properties(self):
        """Get the properties of the underlying algorithm. These are:

        * Can the algorithm handle missing values
          (handles_missing_values : {True, False})
        * Can the algorithm handle nominal features
          (handles_nominal_features : {True, False})
        * Can the algorithm handle numerical features
          (handles_numerical_features : {True, False})
        * Can the algorithm handle multiclass-classification problems
          (handles_multiclass : {True, False})
        * Can preprocess classification data
          (handles_classification_data : {True, False}

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    def get_hyperparameter_search_space(self):
        """Return the configuration space of this preprocessing algorithm.

        Returns
        -------
        HPOlibConfigspace.configuration_space.ConfigurationSpace
            The configuration space of this preprocessing algorithm.
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
        """The transform function calls the transform function of the
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
        return self.preprocessor

    def __str__(self):
        raise NotImplementedError()
