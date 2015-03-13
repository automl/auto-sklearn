
import numpy as np

from HPOlibConfigSpace.forbidden import ForbiddenAndConjunction
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause

from ParamSklearn.util import SPARSE, DENSE, INPUT


def get_match_array(preprocessors, estimators, sparse, pipeline):
    # Now select combinations that work
    # We build a binary matrix, where a 1 indicates, that a combination
    # work on this dataset based in the dataset and the input/output formats
    # A 'zero'-row (column) is an unusable preprocessor (classifier)
    # A single zero results in an forbidden condition
    preprocessors_list = preprocessors.keys()
    estimator_list = estimators.keys()
    matches = np.zeros([len(preprocessors), len(estimators)])
    for pidx, p in enumerate(preprocessors_list):
        p_in = preprocessors[p].get_properties()['input']
        p_out = preprocessors[p].get_properties()['output']
        if p in pipeline:
            continue
        elif sparse and SPARSE not in p_in:
            continue
        elif not sparse and DENSE not in p_in:
            continue
        for cidx, c in enumerate(estimator_list):
            c_in = estimators[c].get_properties()['input']
            if p_out == INPUT:
                # Preprocessor does not change the format
                if (sparse and SPARSE in c_in) or \
                        (not sparse and DENSE in c_in):
                    # Estimator input = Dataset format
                    matches[pidx, cidx] = 1
                    continue
                else:
                    # These won't work
                    continue
            elif p_out == DENSE and DENSE in c_in:
                matches[pidx, cidx] = 1
                continue
            elif p_out == SPARSE and SPARSE in c_in:
                matches[pidx, cidx] = 1
                continue
            else:
                # These won't work
                continue
    return matches


def _get_idx_to_keep(m):
    # Returns all rows and cols where matches contains not only zeros
    keep_row = [idx for idx in range(m.shape[0]) if np.sum(m[idx, :]) != 0]
    keep_col = [idx for idx in range(m.shape[1]) if np.sum(m[:, idx]) != 0]
    return keep_col, keep_row


def sanitize_arrays(m, preprocessors_list, estimators_list,
                    preprocessors, estimators):
    assert len(preprocessors_list) == len(preprocessors.keys())
    assert len(estimators_list) == len(estimators.keys())
    assert isinstance(m, np.ndarray)
    # remove components that are not usable for this problem
    keep_col, keep_row = _get_idx_to_keep(m)

    m = m[keep_row, :]
    m = m[:, keep_col]
    preproc_list = [preprocessors_list[p] for p in keep_row]
    est_list = [estimators_list[p] for p in keep_col]

    new_est = dict()
    for c in est_list:
        new_est[c] = estimators[c]
    new_preproc = dict()
    for p in preproc_list:
        new_preproc[p] = preprocessors[p]

    assert len(new_preproc) == m.shape[0]
    assert len(new_est) == m.shape[1]
    return m, preproc_list, est_list, new_preproc, new_est


def add_forbidden(conf_space, preproc_list, est_list, matches, est_type='classifier'):
    assert est_type in ('classifier', 'regressor'), "'task_type is %s" % est_type

    for pdx, p in enumerate(preproc_list):
        if np.sum(matches[pdx, :]) == matches.shape[1]:
            continue
        for cdx, c in enumerate(est_list):
            if matches[pdx, cdx] == 0:
                try:
                    conf_space.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(conf_space.get_hyperparameter(
                            est_type), c),
                        ForbiddenEqualsClause(conf_space.get_hyperparameter(
                            "preprocessor"), p)))
                except:
                    pass
    return conf_space
