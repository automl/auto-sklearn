import dask.distributed

from distributed.utils_test import inc

import pytest

from autosklearn.util.single_thread_client import SingleThreadedClient


def test_single_thread_client_like_dask_client():
    single_thread_client = SingleThreadedClient()
    assert isinstance(single_thread_client, dask.distributed.Client)
    future = single_thread_client.submit(inc, 1)
    assert isinstance(future, dask.distributed.Future)
    assert future.done()
    assert future.result() == 2
    assert sum(single_thread_client.nthreads().values()) == 1
    single_thread_client.close()
    single_thread_client.shutdown()

    # Client/Futures are printed, so make sure str works
    # str calls __rpr__ which is the purpose of below check
    assert str(future) != ""
    assert str(single_thread_client) != ""

    with pytest.raises(NotImplementedError):
        single_thread_client.get_scheduler_logs()
