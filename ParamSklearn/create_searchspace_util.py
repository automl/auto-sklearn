import numpy as np

from HPOlibConfigSpace.forbidden import ForbiddenAndConjunction
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause

from ParamSklearn.util import SPARSE, DENSE, INPUT


def get_match_array(node_0, node_1, dataset_properties,
                    node_0_include=None, node_0_exclude=None,
                    node_1_include=None, node_1_exclude=None):
    # Select combinations of nodes that work
    # Three cases possible:
    # * node_0 and node_1 are both nodes:
    #   Check if they fit together, return a (1, 1) array
    # * node_0 is a node, node_1 is a composite of nodes (or vice versa)
    #   Check if they fit together, return a (1, n) array
    # * node_0 and node_1 are both composites of nodes
    #   Check if they fit together, return a (n, m) array
    #
    # We build a binary array, where a 1 indicates, that a combination
    # works on this dataset based on the dataset and the input/output formats
    #
    # A 'zero'-row (column) is an unusable preprocessor (classifier)
    # A single zero results in an forbidden condition

    # Duck typing, not sure if it's good...
    sparse = dataset_properties.get('sparse')

    node_0_is_choice = hasattr(node_0, "get_available_components")
    node_1_is_choice = hasattr(node_1, "get_available_components")

    if node_0_is_choice:
        node_0_choices = node_0.get_available_components(
            dataset_properties, include=node_0_include, exclude=node_0_exclude).values()
    else:
        node_0_choices = [node_0]
    if node_1_is_choice:
        node_1_choices = node_1.get_available_components(
            dataset_properties, include=node_1_include, exclude=node_1_exclude).values()
    else:
        node_1_choices = [node_1]

    matches = np.zeros([len(node_0_choices), len(node_1_choices)])

    for n0_idx, n0 in enumerate(node_0_choices):
        if node_0_is_choice and node_0 == n0:
            continue

        node0_in = node_0_choices[n0_idx].get_properties()['input']
        node0_out = node_0_choices[n0_idx].get_properties()['output']

        if sparse and SPARSE not in node0_in:
            continue
        elif not sparse and DENSE not in node0_in:
            continue

        for n1_idx, n1 in enumerate(node_1_choices):
            if node_1_is_choice and node_1 == n1:
                continue

            node1_in = n1.get_properties()['input']
            if node0_out == INPUT:
                # Preprocessor does not change the format
                if (sparse and SPARSE in node1_in) or \
                        (not sparse and DENSE in node1_in):
                    # Estimator input = Dataset format
                    matches[n0_idx, n1_idx] = 1
                else:
                    # These won't work
                    pass
            elif node0_out == DENSE and DENSE in node1_in:
                matches[n0_idx, n1_idx] = 1
            elif node0_out == SPARSE and SPARSE in node1_in:
                matches[n0_idx, n1_idx] = 1
            else:
                # These won't work
                pass
    return matches


def _get_idx_to_keep(matches):
    # Returns all rows and cols where matches contains not only zeros
    keep_row = [idx for idx in range(matches.shape[0]) if np.sum(matches[idx, :]) != 0]
    keep_col = [idx for idx in range(matches.shape[1]) if np.sum(matches[:, idx]) != 0]
    return keep_col, keep_row


def sanitize_arrays(matches, node_0, node_1, dataset_properties,
                    node_0_include=None, node_0_exclude=None,
                    node_1_include=None, node_1_exclude=None):
    node_0_is_choice = hasattr(node_0, "get_available_components")
    node_1_is_choice = hasattr(node_1, "get_available_components")

    if not node_0_is_choice:
        node_0 = [node_0]
    else:
        node_0 = node_0.get_available_components(dataset_properties,
                                                 include=node_0_include,
                                                 exclude=node_0_exclude).keys()
    if not node_1_is_choice:
        node_1 = [node_1]
    else:
        node_1 = node_1.get_available_components(dataset_properties,
                                                 include=node_1_include,
                                                 exclude=node_1_exclude).keys()

    assert matches.shape[0] == len(node_0), (matches.shape[0], len(node_0))
    assert matches.shape[1] == len(node_1), (matches.shape[1], len(node_1))
    assert isinstance(matches, np.ndarray)
    # remove components that are not usable for this problem
    keep_col, keep_row = _get_idx_to_keep(matches)

    matches = matches[keep_row, :]
    matches = matches[:, keep_col]

    node_0_list = [node_0[p] for p in keep_row]
    node_1_list = [node_1[p] for p in keep_col]

    assert len(node_0_list) == matches.shape[0]
    assert len(node_1_list) == matches.shape[1]
    return matches, node_0_list, node_1_list


def add_forbidden(conf_space, node_0_list, node_1_list, matches,
                  node_0_name, node_1_name):
    for pdx, p in enumerate(node_0_list):
        if np.sum(matches[pdx, :]) == matches.shape[1]:
            continue
        for cdx, c in enumerate(node_1_list):
            if matches[pdx, cdx] == 0:
                conf_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(conf_space.get_hyperparameter(
                        node_1_name), c),
                    ForbiddenEqualsClause(conf_space.get_hyperparameter(
                        node_0_name), p)))
    return conf_space
