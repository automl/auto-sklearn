"""Configuration for benchmark tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-save",
        action="store_true",
        default=False,
        help="Save benchmark results to .benchmarks/",
    )
    parser.addoption(
        "--benchmark-compare",
        action="store",
        default=None,
        help="Compare with previous benchmark run (provide commit hash)",
    )


@pytest.fixture(scope="session")
def benchmark_config(request):
    return {
        "save": request.config.getoption("--benchmark-save"),
        "compare": request.config.getoption("--benchmark-compare"),
    }
