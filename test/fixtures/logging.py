from pytest_cases import fixture

from test.mocks.logging import MockLogger


@fixture
def mock_logger() -> MockLogger:
    """A mock logger with some mock defaults"""
    return MockLogger()
