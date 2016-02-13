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
        # http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def convert_multioutput_multiclass_to_multilabel(probas):
    if isinstance(probas, np.ndarray) and len(probas.shape) > 2:
        raise ValueError('New unsupported sklearn output!')
    if isinstance(probas, list):
        multioutput_probas = np.ndarray((probas[0].shape[0], len(probas)))
        for i, output in enumerate(probas):
            if output.shape[1] > 2:
                raise ValueError('Multioutput-Multiclass supported by '
                                 'scikit-learn, but not by auto-sklearn!')
            # Only copy the probability of something having class 1
            elif output.shape[1] == 2:
                multioutput_probas[:, i] = output[:, 1]
            # This label was never observed positive in the training data,
            # therefore it is only the probability for the label being False
            else:
                multioutput_probas[:, i] = 0
        probas = multioutput_probas
    return probas