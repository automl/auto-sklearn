import numpy as np


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        # Compute the Softmax like it is described here:
        # https://www.deeplearningbook.org/contents/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def convert_multioutput_multiclass_to_multilabel(probas):
    """Converts the model predicted probabilities to useable format.

    In some cases, models predicted_proba can output an array of shape
    (2, n_samples, n_labels) where the 2 stands for the probability of positive
    or negative. This function will convert this to an (n_samples, n_labels)
    array where the probability of a label being true is kept.

    Parameters
    ----------
    probas: array_like (1 or 2, n_samples, n_labels) or (n_samples, n_labels)
        The output of predict_proba of a classifier model

    Returns
    -------
    np.ndarray shape of (n_samples, n_labels)
        The probabilities of each label for every sample
    """
    if isinstance(probas, list):
        # probas = shape of (prob of positive/negative label, n_samples, n_labels)
        n_samples = probas[0].shape[0]
        n_labels = len(probas)
        multioutput_probas = np.ndarray((n_samples, n_labels))

        for i, label_scores in enumerate(probas):
            n_probabilities = label_scores.shape[1]

            # Label was never observed so it has a probability of 0
            if n_probabilities == 1:
                multioutput_probas[:, i] = 0

            # Label has a probability score for true or false
            elif n_probabilities == 2:
                multioutput_probas[:, i] = label_scores[:, 1]

            # In case multioutput-multiclass input was used, where we have
            # a probability for each class
            elif n_probabilities > 2:
                raise ValueError(
                    "Multioutput-Multiclass supported by "
                    "scikit-learn, but not by auto-sklearn!"
                )
            else:
                RuntimeError(f"Unkown predict_proba output={probas}")

        return multioutput_probas

    elif isinstance(probas, np.ndarray):
        if len(probas.shape) > 2:
            raise ValueError("New unsupported sklearn output!")
        else:
            return probas

    else:
        return ValueError(f"Unrecognized probas\n{type(probas)}\n{probas}")
