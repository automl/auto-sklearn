from pathlib import Path

from dask.distributed import Client, LocalCluster

from autosklearn.util.dask import LocalDask, UserDask

import pytest


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_user_dask(tmp_path: Path, n_jobs: int) -> None:
    """
    Expects
    -------
    * A UserDask should not close the client after exiting context
    """
    cluster = LocalCluster(
        n_workers=n_jobs,
        processes=False,
        threads_per_worker=1,
        local_directory=tmp_path,
    )
    client = Client(cluster, heartbeat_interval=10000)

    # Active at creation
    dask = UserDask(client)

    client_1 = None
    with dask as user_client:
        client_1 = user_client
        assert user_client.status == "running"

    client_2 = None
    with dask as user_client:
        assert user_client.status == "running"
        client_2 = user_client

    # Make sure they are the same client
    assert id(client_1) == id(client_2)

    # Remains running after context
    assert client_1.status == "running"

    cluster.close()
    client.close()

    assert client.status == "closed"


def test_local_dask_creates_new_clients(tmp_path: Path) -> None:
    """
    Expects
    -------
    * A LocalDask should create new dask clusters at each context usage
    """
    # We need 2 to use an actual dask client and not a SingleThreadedClient
    local_dask = LocalDask(n_jobs=2)

    client_1 = None
    with local_dask as client:
        client_1 = client
        assert client_1.status == "running"

    assert client_1.status == "closed"

    client_2 = None
    with local_dask as client:
        client_2 = client
        assert client_2.status == "running"

    # Make sure they were different clients
    assert id(client_1) != id(client_2)

    assert client_2.status == "closed"
    assert client_1.status == "closed"
