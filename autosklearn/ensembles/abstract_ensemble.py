from abc import ABCMeta, abstractmethod


class AbstractEnsemble(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, base_models_predictions, true_targets, model_identifiers):
        """Fit an ensemble given predictions of base models and targets.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        Returns
        -------
        self

        """
        pass

    @abstractmethod
    def predict(self, base_models_predictions):
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            Same as in the fit method.

        Returns
        -------
        array : [n_data_points]
        """
        self

    @abstractmethod
    def pprint_ensemble_string(self, models):
        """Return a nicely-readable representation of the ensmble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        str
        """

    @abstractmethod
    def get_model_identifiers(self):
        """Return identifiers of models in the ensemble.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
