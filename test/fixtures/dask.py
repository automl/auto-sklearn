from __future__ import annotations

from typing import Callable

from dask.distributed import Client, get_client

from pytest import FixtureRequest, fixture

# Terrible practice but we need to close dask clients somehow
active_clients: dict[str, Callable] = {}


@fixture(autouse=True)
def clean_up_any_dask_clients(request: FixtureRequest) -> None:
    """Auto injected fixture to close dask clients after each test"""
    yield
    if any(active_clients):
        for adr in list(active_clients.keys()):
            if request.config.getoption("verbose") > 1:
                print(f"\nFixture closing dask_client at {adr}")

            close = active_clients[adr]
            close()
            del active_clients[adr]


def create_test_dask_client(n_workers: int = 2) -> Client:
    """Factory to make a Dask client and a function to close it
    them

    Parameters
    ----------
    n_workers: int = 2
        inside asklea
        inside AutoML.

    Returns
    -------
    Client, Callable
        The client and a function to call to close that client
    """
    # Workers are in subprocesses to not create deadlocks with the pynisher
    # and logging.
    client = Client(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=False,
        scheduler_port=0,  # Set to 0 so it chooses a random one
        dashboard_address=None,  # Disable dashboarding
    )
    adr = client.scheduler_info()["address"]

    def close() -> None:
        try:
            client = get_client(adr, timeout=1)
            client.shutdown()
        except Exception:
            pass

    active_clients[adr] = close

    return client


@fixture
def make_dask_client() -> Callable[[int], Client]:
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
    return create_test_dask_client


# TODO remove in favour of make_dask_client
@fixture(scope="function")
def dask_client(make_dask_client: Callable) -> Client:
    """Create a dask client with two workers."""
    client = make_dask_client(n_workers=2)
    yield client


# TODO remove in favour of make_dask_client
@fixture(scope="function")
def dask_client_single_worker(make_dask_client: Callable) -> Client:
    """Dask client with only 1 worker

    Note
    ----
    May create deadlocks with logging and pynisher
    """
    client = make_dask_client(n_workers=1)
    yield client
