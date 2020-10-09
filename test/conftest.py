import os
import shutil
import time

import dask
from dask.distributed import Client, get_client
import psutil
import pytest

from autosklearn.util.backend import create


@pytest.fixture(scope="function")
def backend(request):

    test_dir = os.path.dirname(__file__)
    tmp = os.path.join(test_dir, '.tmp__%s__%s' % (request.module.__name__, request.node.name))
    output = os.path.join(
        test_dir, '.output__%s__%s' % (request.module.__name__, request.node.name)
    )

    for dir in (tmp, output):
        for i in range(10):
            if os.path.exists(dir):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    # Make sure the folders we wanna create do not already exist.
    backend = create(
        tmp,
        output,
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
    )

    def get_finalizer(tmp_dir, output_dir):
        def session_run_at_end():
            for dir in (tmp_dir, output_dir):
                for i in range(10):
                    if os.path.exists(dir):
                        try:
                            shutil.rmtree(dir)
                            break
                        except OSError:
                            time.sleep(1)
        return session_run_at_end
    request.addfinalizer(get_finalizer(tmp, output))

    return backend


@pytest.fixture(scope="function")
def tmp_dir(request):
    return _dir_fixture('tmp', request)


@pytest.fixture(scope="function")
def output_dir(request):
    return _dir_fixture('output', request)


def _dir_fixture(dir_type, request):

    test_dir = os.path.dirname(__file__)
    dir = os.path.join(
        test_dir, '.%s__%s__%s' % (dir_type, request.module.__name__, request.node.name)
    )

    for i in range(10):
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                break
            except OSError:
                pass

    def get_finalizer(dir):
        def session_run_at_end():
            for i in range(10):
                if os.path.exists(dir):
                    try:
                        shutil.rmtree(dir)
                        break
                    except OSError:
                        time.sleep(1)

        return session_run_at_end

    request.addfinalizer(get_finalizer(dir))


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
    dask.config.set({'distributed.worker.daemon': False})

    client = Client(n_workers=2, threads_per_worker=1, processes=False)
    print("Started Dask client={}\n".format(client))

    def get_finalizer(address):
        def session_run_at_end():
            client = get_client(address)
            print("Closed Dask client={}\n".format(client))
            client.shutdown()
            client.close()
            del client
        return session_run_at_end
    request.addfinalizer(get_finalizer(client.scheduler_info()['address']))

    return client


def pytest_sessionfinish(session, exitstatus):
    proc = psutil.Process()
    for child in proc.children(recursive=True):
        print(child, child.cmdline())
