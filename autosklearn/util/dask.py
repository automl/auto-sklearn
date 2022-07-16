""" Provides simplified 2 use cases of dask that we consider

1. A UserDask is when a user supplies a dask client, in which case
we don't close this down and leave it up to the user to control its lifetime.
2.  A LocalDask is one we use when no user dask is supplied. In this case
we make sure to spin up and close down clients as needed.

Both of these can be uniformly accessed as a context manager.

.. code:: python

    # Locally controlled dask client
    local_dask = LocalDask(n_jobs=2)
    with local_dask as client:
        # Do stuff with client
        ...

    # `client` is shutdown properly

    # ----------------

    # User controlled dask client
    user_dask = UserDask(user_client)

    with user_dask as client:
        # Do stuff with (client == user_client)
        ...

    # `user_client` is still open and up to the user to close
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import tempfile

from dask.distributed import Client, LocalCluster

from autosklearn.util.single_thread_client import SingleThreadedClient


class Dask(ABC):
    @abstractmethod
    def client(self) -> Client:
        """Should return a dask client"""
        ...

    @abstractmethod
    def close(self) -> None:
        """Should close up any resources needed for the dask client"""
        ...

    def __enter__(self) -> Client:
        return self.client()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    @abstractmethod
    def __repr__(self) -> str:
        ...


class UserDask(Dask):
    """A dask instance created by a user"""

    def __init__(self, client: Client):
        """
        Parameters
        ----------
        client : Client
            The client they passed in
        """
        self._client = client

    def client(self) -> Client:
        """The dask client"""
        return self._client

    def close(self) -> None:
        """Close the dask client"""
        # We do nothing, it's user provided
        pass

    def __repr__(self) -> str:
        return "UserDask(...)"


class LocalDask(Dask):
    def __init__(self, n_jobs: int | None = None) -> None:
        self.n_jobs = n_jobs
        self._client: Client | None = None
        self._cluster: LocalCluster | None = None

    def client(self) -> Client:
        """Creates a usable dask client or returns an existing one

        If there is not current client, because it has been closed, create
        a new one.
        * If ``n_jobs == 1``, create a ``SingleThreadedClient``
        * Else create a ``Client`` with a ``LocalCluster``
        """
        if self._client is not None:
            return self._client

        if self.n_jobs == 1:
            cluster = None
            client = SingleThreadedClient()
        else:
            cluster = LocalCluster(
                n_workers=self.n_jobs,
                processes=False,
                threads_per_worker=1,
                # We use tmpdir to save the workers as deleting workers takes
                # more time than deleting backend directories.
                # This prevent an error saying that the worker file was deleted,
                # so the client could not close the worker properly
                local_directory=tempfile.gettempdir(),
                # Memory is handled by the pynisher, not by the dask worker/nanny
                memory_limit=0,
            )
            client = Client(cluster, heartbeat_interval=10000)  # 10s

        self._client = client
        self._cluster = cluster
        return self._client

    def close(self) -> None:
        """Closes any open dask client"""
        if self._client is None:
            return

        self._client.close()
        if self._cluster is not None:
            self._cluster.close()

        self._client = None
        self._cluster = None

    def __repr__(self) -> str:
        return f"LocalDask(n_jobs = {self.n_jobs})"
