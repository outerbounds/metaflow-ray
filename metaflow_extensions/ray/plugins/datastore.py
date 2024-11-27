from typing import List, Dict, Union, Tuple
import time
import sys
import os
from io import BytesIO
from collections import namedtuple
from functools import partial
from contextlib import contextmanager
from .exceptions import DatastoreKeyNotFoundError, BarrierTimeoutException
from .constants import RAY_SUFFIX

# mimic a subset of the behavior of the Metaflow S3Object
DatastoreBlob = namedtuple("DatastoreBlob", "blob url text")
ListPathResult = namedtuple("ListPathResult", "url")


class DecoratorDatastore(object):

    def __init__(self, flow_datastore, pathspec, attempt):
        self._backend = flow_datastore._storage_impl
        self._flow_name = flow_datastore.flow_name
        _, run_id, step_name, _ = pathspec.split("/")
        self._run_id = run_id
        self._step_name = step_name
        self._attempt = str(attempt)

    @property
    def get_storage_root(self):
        """
        Return the path to the root of the datastore.
        This method is where the unique datastore root for each cloud provider is specified.

        Note: S3Storage class uses the S3 client (other clouds do not have this),
            which prepends the storage root inside the self._backend calls this class uses.
        """
        if self._backend.TYPE == "s3":
            return RAY_SUFFIX
        elif self._backend.TYPE == "azure":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE

            return os.path.join(DATASTORE_SYSROOT_AZURE, RAY_SUFFIX)
        elif self._backend.TYPE == "gs":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_GS

            return os.path.join(DATASTORE_SYSROOT_GS, RAY_SUFFIX)
        else:
            raise NotImplementedError(
                "Datastore does not support backend %s" % (self._backend.TYPE)
            )

    def get_datastore_key_location(self):
        return os.path.join(
            self.get_storage_root,
            self._flow_name,
            self._run_id,
            self._step_name,
            self._attempt,
        )

    def get_datastore_file_location(self, key):
        return os.path.join(self.get_datastore_key_location(), key)

    def put(self, key: str, obj: Union[str, bytes], overwrite: bool = False):
        "Put a single object into the datastore's `key` index."
        _save_object = None
        if isinstance(obj, bytes):
            _save_object = BytesIO(obj)
        else:
            _save_object = BytesIO(obj.encode("utf-8"))

        self._backend.save_bytes(
            [(self.get_datastore_file_location(key), _save_object)],
            overwrite=overwrite,
        )

    def put_files(self, key_paths: List[Tuple[str, str]], overwrite=False):
        results = []
        for key, path in key_paths:
            with open(path, "rb") as f:
                self.put(key, f.read(), overwrite=overwrite)
            results.append(self.get_datastore_file_location(key))
        return results

    def get(self, key):
        "Get a single object residing in the datastore's `key` index."
        datastore_url = self.get_datastore_file_location(key)
        with self._backend.load_bytes([datastore_url]) as get_results:
            for key, path, _ in get_results:
                if path is not None:
                    with open(path, "rb") as f:
                        blob_bytes = f.read()
                        return DatastoreBlob(
                            blob=blob_bytes,
                            url=datastore_url,
                            text=blob_bytes.decode("utf-8"),
                        )
                else:
                    raise DatastoreKeyNotFoundError(datastore_url)

    def get_many(self, keys):
        return [self.get(key) for key in keys]

    def list_paths(self, keys):
        "List all objects in the datastore's `keys` index."
        keys = [self.get_datastore_file_location(key) for key in keys]
        list_path_results = [
            ListPathResult(url=list_content_result.path)
            for list_content_result in self._backend.list_content(keys)
        ]
        return list_path_results


def _best_effort_read_key(datastore: DecoratorDatastore, key: str):
    """
    Read a key from the datastore, but do not raise an error if the key is not found.
    """
    try:
        return datastore.get(key)
    except DatastoreKeyNotFoundError:
        return None


def _best_effort_get_keys(datastore: DecoratorDatastore, keys: List[str]):
    """
    Get the keys from the datastore, but do not raise an error if the keys are not found.
    """
    results = {}
    not_found_keys = []
    for key in keys:
        data = _best_effort_read_key(datastore, key)
        if data:
            results[key] = data
        else:
            not_found_keys.append(key)
    return results, not_found_keys


def _warning_logger(barrier_name, msg):
    print("[%s] %s" % (barrier_name, msg), file=sys.stderr)


def wait_for_key_data(
    datastore: DecoratorDatastore,
    keys: List[str],
    max_wait_time: float = 600,
    frequency=5,
    logger=None,
    wait_message=None,
) -> Dict[str, DatastoreBlob]:
    """
    Wait for the keys to be available in the datastore.
    If the keys are not available after `max_wait_time` seconds, raise an error.
    """
    start = time.time()
    exit_condition = lambda: time.time() - start > max_wait_time
    _current_keys = keys.copy()
    main_data = {}
    _iter = 0
    while not exit_condition():
        data, _ = _best_effort_get_keys(datastore, _current_keys)
        # if all keys are found, return the data
        if len(main_data) == len(keys):
            return main_data
        main_data.update(data)
        # update the current keys to wait for the remaining keys
        _current_keys = list(set(keys) - set(main_data.keys()))
        if logger is not None and wait_message is not None and _iter % 10 == 0:
            logger(wait_message)
        _iter += 1
        time.sleep(frequency)

    raise DatastoreKeyNotFoundError(
        f"Keys {keys} were not found in the datastore after {max_wait_time} seconds."
    )


def task_sync_barrier(
    barrier_name,
    datastore: DecoratorDatastore,
    keys: List[str],
    max_wait_time=600,
    frequency=5,
    description=None,
    wait_message=None,
):
    """
    A barrier that waits for keys to be written to the datastore and acts like a distributed-barrier.
    When multiple tasks are running in parallel, this barrier can be used to ensure that all tasks
    can wait on a certain keys to be written to the datastore. If the keys are not written to the datastore
    after `max_wait_time` seconds, a `BarrierTimeoutException` error is raised. This way only once all the keys
    are written to the datastore, the tasks will proceed. This barrier is used to ensure that all tasks
    are in sync before proceeding to the next step.

    Args:
        barrier_name (str): The name of the barrier. Used for debugging purposes.
        datastore (DecoratorDatastore)
        keys (List[str]): The keys to wait for in the datastore.
        max_wait_time (float): The maximum time to wait for the keys to be written to the datastore.
        frequency (float): The frequency to check the datastore for the keys.
        description (str): A description of the barrier. Used for debugging purposes.
        wait_message (str): A message to show when the barrier is waiting for the keys to be written to the datastore.
    """
    try:
        data = wait_for_key_data(
            datastore,
            keys,
            max_wait_time,
            frequency,
            logger=partial(_warning_logger, barrier_name),
            wait_message=wait_message,
        )
    except DatastoreKeyNotFoundError:
        raise BarrierTimeoutException(barrier_name, description)
    else:
        return data
