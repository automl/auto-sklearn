# content of conftest.py

# we define a fixture function below and it will be "used" by
# referencing its name from tests

import pytest

@pytest.fixture(scope="class")
def db_class(request):
    class DummyDB:
        pass
    # set a class attribute on the invoking test context
    request.cls.db = DummyDB()

@pytest.fixture(scope="session", autouse=True)
def my_own_session_run_at_beginning(request):
    import dask
    dask.config.set({'distributed.worker.daemon': False})
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2, scheduler_port=4567)
    client = Client(cluster)
    print(f"Started Dask client={client}\n")

    def my_own_session_run_at_end():
        from dask.distributed import get_client
        client = get_client('127.0.0.1:4567')
        print(f"\nClosing client={client}\n")
        try:
            client.shutdown()
            client.close(timeout=5)
        except Exception as e:
            print(f"Error while closing the client={e}")
    request.addfinalizer(my_own_session_run_at_end)
