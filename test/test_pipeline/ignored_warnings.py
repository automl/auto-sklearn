from contextlib import contextmanager
from typing import List, Iterator, Tuple

import warnings

from sklearn.exceptions import ConvergenceWarning


regressor_warnings = [
    (
        UserWarning, (  # From QuantileTransformer
            r"n_quantiles \(\d+\) is greater than the total number of samples \(\d+\)\."
            r" n_quantiles is set to n_samples\."
        )
    ),
    (
        ConvergenceWarning, (  # From GaussianProcesses
            r"The optimal value found for dimension \d+ of parameter \w+ is close"
            r" to the specified (upper|lower) bound .*(Increasing|Decreasing) the bound"
            r" and calling fit again may find a better value."
        )
    ),
    (
        UserWarning, (  # From FastICA
            r"n_components is too large: it will be set to \d+"
        )
    ),
    (
        ConvergenceWarning, (  # From SGD
            r"Maximum number of iteration reached before convergence\. Consider increasing"
            r" max_iter to improve the fit\."
        )
    ),
    (
        ConvergenceWarning, (  # From MLP
            r"Stochastic Optimizer: Maximum iterations \(\d+\) reached and the"
            r" optimization hasn't converged yet\."
        )
    ),
]

classifier_warnings = [
    (
        UserWarning, (  # From QuantileTransformer
            r"n_quantiles \(\d+\) is greater than the total number of samples \(\d+\)\."
            r" n_quantiles is set to n_samples\."
        )
    ),
    (
        UserWarning, (  # From FastICA
            r"n_components is too large: it will be set to \d+"
        )

    ),
    (
        ConvergenceWarning, (  # From Liblinear
            r"Liblinear failed to converge, increase the number of iterations\."
        )
    ),
    (
        ConvergenceWarning, (  # From SGD
            r"Maximum number of iteration reached before convergence\. Consider increasing"
            r" max_iter to improve the fit\."
        )
    ),
    (
        ConvergenceWarning, (  # From MLP
            r"Stochastic Optimizer: Maximum iterations \(\d+\) reached and the"
            r" optimization hasn't converged yet\."
        )
    ),
    (
        ConvergenceWarning, (  # From FastICA
            r"FastICA did not converge\."
            r" Consider increasing tolerance or the maximum number of iterations\."
        )
    ),
    (
        UserWarning, (  # From LDA (Linear Discriminant Analysis)
            r"Variables are collinear"
        )
    ),
    (
        UserWarning, (
            r"Clustering metrics expects discrete values but received continuous values"
            r" for label, and multiclass values for target"
        )
    )
]

feature_preprocessing_warnings = [
    (
        ConvergenceWarning, (  # From liblinear
            r"Liblinear failed to converge, increase the number of iterations."
        )
    )
]

ignored_warnings = regressor_warnings + classifier_warnings + feature_preprocessing_warnings


@contextmanager
def ignore_warnings(to_ignore: List[Tuple[Exception, str]] = ignored_warnings) -> Iterator[None]:
    """A context manager to ignore warnings

    >>> with ignore_warnings(classifier_warnings):
    >>>     ...

    Parameters
    ----------
    to_ignore: List[Tuple[Exception, str]] = ignored_warnings
        The list of warnings to ignore, defaults to all registered warnings
    """
    with warnings.catch_warnings():
        for category, message in to_ignore:
            warnings.filterwarnings('ignore', category=category, message=message)
        yield
