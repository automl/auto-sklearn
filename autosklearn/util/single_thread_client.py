import typing
from pathlib import Path

import dask.distributed


class DummyFuture(dask.distributed.Future):
    """
    A class that mimics a distributed Future, the outcome of
    performing submit on a distributed client.
    """
    def __init__(self, result):
        self._result = result  # type: typing.Any

    def result(self, timeout: typing.Optional[int] = None) -> typing.Any:
        return self._result

    def cancel(self) -> None:
        pass

    def done(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "DummyFuture: {}".format(self._result)

    def __del__(self) -> None:
        pass


class SingleThreadedClient(dask.distributed.Client):
    """
    A class to Mock the Distributed Client class, in case
    Auto-Sklearn is meant to run in the current Thread.
    """
    def __init__(self):
        pass

    def submit(
        self,
        func: typing.Callable,
        *args,
        priority: int = 0,
        **kwargs,
    ) -> typing.Any:
        return DummyFuture(func(*args, **kwargs))

    def close(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def write_scheduler_file(self, scheduler_file: str) -> None:
        Path(scheduler_file).touch()
        return

    def _get_scheduler_info(self) -> typing.Dict:
        return {
            'workers': ['127.0.0.1'],
            'type': 'Scheduler',
        }

    def nthreads(self) -> typing.Dict:
        return {
            '127.0.0.1': 1,
        }

    def __repr__(self) -> str:
        return 'SingleThreadedClient()'

    def __del__(self) -> None:
        pass
