import itertools

import numpy as np

from HPOlibConfigSpace.forbidden import ForbiddenAndConjunction
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause

from ParamSklearn.util import SPARSE, DENSE, INPUT, PREDICTIONS


def get_match_array(pipeline, dataset_properties, include=None,
                                   exclude=None):
    sparse = dataset_properties.get('sparse')

    # Duck typing, not sure if it's good...
    node_i_is_choice = []
    node_i_choices = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices.append(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).values())

        else:
            node_i_choices.append([node])

    matches_dimensions = [len(choices) for choices in node_i_choices]
    # Start by allowing every combination of nodes. Go through all
    # combinations/pipelines and erase the illegal ones
    matches = np.ones(matches_dimensions, dtype=int)

    pipeline_idxs = [range(dim) for dim in matches_dimensions]
    for pipeline_instantiation_idxs  in itertools.product(*pipeline_idxs):
        pipeline_instantiation = [node_i_choices[i][idx] for i, idx in
                                  enumerate(pipeline_instantiation_idxs)]

        data_is_sparse = sparse
        for node in pipeline_instantiation:
            node_input = node.get_properties()['input']
            node_output = node.get_properties()['output']

            if (data_is_sparse and SPARSE not in node_input) or \
                    not data_is_sparse and DENSE not in node_input:
                matches[pipeline_instantiation_idxs] = 0
                break

            if INPUT in node_output or PREDICTIONS in node_output or\
                    (not data_is_sparse and DENSE in node_input and
                        node_output == DENSE) or \
                    (data_is_sparse and SPARSE in node_input and node_output
                        == SPARSE):
                # Don't change the data_is_sparse flag
                pass
            elif data_is_sparse and DENSE in node_output:
                data_is_sparse = False
            elif not data_is_sparse and SPARSE in node_output:
                data_is_sparse = True
            else:
                print node
                print data_is_sparse
                print node_input, node_output
                raise ValueError("This combination is not allowed!")

    return matches


def find_active_choices(matches, node, node_idx, dataset_properties, \
                        include=None, exclude=None):
    if not hasattr(node, "get_available_components"):
        raise ValueError()
    available_components = node.get_available_components(dataset_properties,
                                                         include=include,
                                                         exclude=exclude)
    assert matches.shape[node_idx] == len(available_components), \
        (matches.shape[node_idx], len(available_components))

    choices = []
    for c_idx, component in enumerate(available_components):
        slices = [slice(None) if idx != node_idx else slice(c_idx, c_idx+1)
                  for idx in range(len(matches.shape))]

        if np.sum(matches[slices]) > 0:
            choices.append(component)
    return choices


def add_forbidden(conf_space, pipeline, matches, dataset_properties,
                  include, exclude):
    # Not sure if this works for 3D
    node_i_is_choice = []
    node_i_choices = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices.append(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).values())

        else:
            node_i_choices.append([node])

    # Find out all chains of choices. Only in such a chain its possible to
    # have several forbidden constraints
    choices_chains = []
    idx = 0
    while idx < len(pipeline):
        if node_i_is_choice[idx]:
            chain_start = idx
            idx += 1
            while idx < len(pipeline) and node_i_is_choice[idx]:
                idx += 1
            chain_stop = idx
            choices_chains.append((chain_start, chain_stop))
        idx += 1

    for choices_chain in choices_chains:
        constraints = set()
        possible_constraints = set()
        possible_constraints_by_length = dict()

        chain_start = choices_chain[0]
        chain_stop = choices_chain[1]
        chain_length = chain_stop - chain_start

        # Add one to have also have chain_length in the range
        for sub_chain_length in range(2, chain_length + 1):
            if sub_chain_length > 2:
                break

            for start_idx in range(chain_start, chain_stop - sub_chain_length + 1):
                #print chain_start + start_idx, sub_chain_length

                indices = range(start_idx, start_idx + sub_chain_length)
                #print indices

                node_0_idx = indices[0]
                node_1_idx = indices[1]

                node_0_name, node_0 = pipeline[node_0_idx]
                node_1_name, node_1 = pipeline[node_1_idx]
                node_0_is_choice = hasattr(node_0, "get_available_components")
                node_1_is_choice = hasattr(node_1, "get_available_components")

                if not node_0_is_choice or not node_1_is_choice:
                    continue

                # Now iterate all combinations and add them as forbidden!
                for pdx, p in enumerate(node_0.get_available_components(dataset_properties)):
                    slices_0 = [
                        slice(None) if idx != node_0_idx else
                        slice(pdx, pdx + 1) for idx in range(len(matches.shape))]
                    if np.sum(matches[slices_0]) == 0:
                        continue

                    for cdx, c in enumerate(node_1.get_available_components(dataset_properties)):

                        slices_1 = [
                            slice(None) if idx != node_1_idx else
                            slice(cdx, cdx + 1) for idx in range(len(matches.shape))]
                        if np.sum(matches[slices_1]) == 0:
                            continue

                        slices = [slice(None) if idx not in (node_0_idx, node_1_idx)
                                  else slice(pdx if idx is node_0_idx else cdx,
                                             pdx+1 if idx is node_0_idx else cdx+1)
                                  for idx in range(len(matches.shape))]

                        #print node_0_name, node_1_name, p, c, matches[slices]
                        if np.sum(matches[slices]) == 0:
                            conf_space.add_forbidden_clause(ForbiddenAndConjunction(
                                ForbiddenEqualsClause(conf_space.get_hyperparameter(
                                    node_1_name + ":__choice__"), c),
                                ForbiddenEqualsClause(conf_space.get_hyperparameter(
                                    node_0_name + ":__choice__"), p)))
                            constraints.add(((node_0_name, p), (node_1_name, c)))

                        elif np.size(matches[slices]) > np.sum(matches[slices]) > 0:
                            #possible_constraints.add()
                            pass

    return conf_space
