"""Test the output of loading the pareto set from an automl instance"""
from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases
from pytest_cases.filters import has_tag

import test.test_automl.cases as cases

has_ensemble = has_tag("fitted") & ~has_tag("no_ensemble")

single_objective = has_ensemble & ~has_tag("multiobjective")
multi_objective = has_ensemble & has_tag("multiobjective")


@parametrize_with_cases("automl", cases=cases, filter=single_objective)
def test_can_output_pareto_front_singleobjective(automl: AutoML) -> None:
    """
    Expects
    -------
    * Non-multiobjective instances should have a pareto set of size 1
    """
    pareto_set = automl._load_pareto_set()

    assert len(pareto_set) == 1


@parametrize_with_cases("automl", cases=cases, filter=multi_objective)
def test_can_output_pareto_front_multiobjective(automl: AutoML) -> None:
    """
    Expects
    -------
    * Multiobjective ensembles should return >= 1, #TODO should test it's pareto optimal
    """
    pareto_set = automl._load_pareto_set()

    assert len(pareto_set) >= 1
