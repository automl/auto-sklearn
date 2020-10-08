import pytest


@pytest.fixture(scope="function")
def dask_client(request):
    """
    This fixture is meant to be called one per pytest session.

    The goal of this function is to create a global client at the start
    of the testing phase. We can create clients at the start of the
    session (this case, as above scope is session), module, class or function
    level.

    The overhead of creating a dask client per class/module/session is something
    that travis cannot handle, so we rely on the following execution flow:

    1- At the start of the pytest session, session_run_at_beginning fixture is called
    to create a global client on port 4567.
    2- Any test that needs a client, would query the global scheduler that allows
    communication through port 4567.
    3- At the end of the test, we shutdown any remaining work being done by any worker
    in the client. This has a maximum 10 seconds timeout. The client object will afterwards
    be empty and when pytest closes, it can safely delete the object without hanging.

    More info on this file can be found on:
    https://docs.pytest.org/en/stable/writing_plugins.html#conftest-py-plugins
    """
    import dask
    dask.config.set({'distributed.worker.daemon': False})
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
    )
    client = Client(cluster)
    print("Started Dask client={}\n".format(client))

    def get_finalizer(address):
        def session_run_at_end():
            from dask.distributed import get_client
            client = get_client(address)
            print("Closed Dask client={}\n".format(client))
            client.shutdown()
        return session_run_at_end
    request.addfinalizer(get_finalizer(cluster.scheduler_address))

    return client
