import numpy as np

from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer

from smac.runhistory.runhistory import RunHistory


class SingleBest(AbstractEnsemble):
    """
    In the case of a crash, this class searches
    for the best individual model.

    Such model is returned as an ensemble of a single
    object, to comply with the expected interface of an
    AbstractEnsemble.
    """
    def __init__(
        self,
        metric: Scorer,
        random_state: np.random.RandomState = None,
        run_history: RunHistory = None,
    ):
        self.metric = metric
        self.random_state = random_state

        # Add some default values -- at least 1 model in ensemble is assumed
        self.indices_ = [0]
        self.weights_ = [1]
        self.run_history = run_history
        self.identifiers_ = self.get_identifiers_from_run_history()

    def get_identifiers_from_run_history(self):
        """
        This method parses the run history, to identify
        the best performing model

        It populates the identifiers attribute, which is used
        by the backend to access the actual model
        """
        best_model_identifier = None
        best_model_score = self.metric._worst_possible_result

        for run_key in self.run_history.data.keys():
            run_value = self.run_history.data[run_key]
            score = self.metric._optimum - (self.metric._sign * run_value.cost)
            if (score > best_model_score and self.metric._sign > 0) \
                    or (score < best_model_score and self.metric._sign < 0):
                best_model_identifier = [
                    (self.random_state, run_value.additional_info['num_run'], run_key.budget)
                ]
                best_model_score = score

        if best_model_identifier is None:
            raise ValueError(
                "No valid model found in run history. This means smac was not able to fit"
                " a valid model. Please check the log file for errors."
            )

        return best_model_identifier

    def predict(self, predictions):
        return predictions[0]

    def __str__(self):
        return 'Single Model Selection:\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models):
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output
