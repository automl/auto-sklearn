from typing import Callable, Tuple

from dask.distributed import Client, get_client
from pytest import fixture


@fixture
def make_dask_client() -> Callable[[int], Tuple[Client, Callable]]:
    """Factory to make a Dask client and a function to close it

    Parameters
    ----------
    n_workers: int = 1
        How many workers to have in the dask client

    Returns
    -------
    Client, Callable
        The client and a function to call to close that client
    """

    def _make(n_workers: int = 1) -> Client:
        # Workers are in subprocesses to not create deadlocks with the pynisher
        # and logging.
        client = Client(n_workers=n_workers, threads_per_worker=1, processes=False)
        adr = client.scheduler_info()["address"]

        def close() -> None:
            client = get_client(adr)
            client.shutdown()
            client.close()
            del client

        return client, close

    return _make


@fixture(scope="function")
def dask_client(make_dask_client: Callable) -> Client:
    """Create a dask client with two workers."""
    client, close = make_dask_client(n_workers=2)
    yield client
    close()


@fixture(scope="function")
def dask_client_single_worker(make_dask_client: Callable) -> Client:
    """Dask client with only 1 worker

    Note
    ----
    May create deadlocks with logging and pynisher
    """
    client, close = make_dask_client(n_workers=1)
    yield client
    close()
