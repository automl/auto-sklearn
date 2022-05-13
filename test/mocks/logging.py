from typing import Optional

from autosklearn.util.logging_ import PicklableClientLogger

from unittest.mock import Mock

MOCKNAME = "mock"
MOCKHOST = "mockhost"
MOCKPORT = 9020


class MockLogger(PicklableClientLogger):
    """Should not be used for testing the actual loggers functionality

    Overwrites all methods with mock objects that can be queries
    * All logging methods do nothing
    * isEnabledFor returns True for everything as it's part of the logging config we
      don't have access to
    * __setstate__ and __getstate__ remain the same and are not mocked
    """

    def __init__(
        self,
        name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.name = name or MOCKNAME
        self.host = host or MOCKHOST
        self.port = port or MOCKPORT

        # Overwrite the logging implementations with mocks
        self.debug = Mock(return_value=None)  # type: ignore
        self.info = Mock(return_value=None)  # type: ignore
        self.warning = Mock(return_value=None)  # type: ignore
        self.error = Mock(return_value=None)  # type: ignore
        self.exception = Mock(return_value=None)  # type: ignore
        self.critical = Mock(return_value=None)  # type: ignore
        self.log = Mock(return_value=None)  # type: ignore
        self.isEnabledFor = Mock(return_value=True)  # type: ignore
