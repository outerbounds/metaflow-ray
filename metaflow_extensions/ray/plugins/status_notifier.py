import os
import json
import time
from collections import namedtuple
from threading import Thread, Event
from .datastore import DecoratorDatastore, _best_effort_read_key
from .ray_utils import warning_message

DEBUG_MODE = os.environ.get("METAFLOW_RAY_DEBUG_MODE", "false").lower() in [
    "true",
    "1",
    "enabled",
    "enable",
]

TaskStatus = namedtuple("TaskStatus", ["status", "timestamp", "node_index"])


def debug_logger(message):
    if DEBUG_MODE:
        warning_message(message)


class TASK_STATUS:
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    UNREACHABLE = "unreachable"


class TaskStatusNotifier:
    ROOT = "node_status"

    def __init__(
        self,
        datastore: DecoratorDatastore,
    ):
        self._datastore = datastore

    def _path(self, node_index):
        return os.path.join(self.ROOT, str(node_index))

    def _create_status(self, status, node_index):
        return json.dumps(
            {"status": status, "timestamp": time.time(), "node_index": node_index}
        )

    def _notify(self, status: str, node_index: int):
        self._datastore.put(
            self._path(node_index),
            self._create_status(status, node_index),
            overwrite=True,
        )

    def heartbeat(self, node_index: int):
        self._notify(TASK_STATUS.RUNNING, node_index)

    def running(self, node_index: int):
        self._notify(TASK_STATUS.RUNNING, node_index)

    def finished(self, node_index: int):
        self._notify(TASK_STATUS.FINISHED, node_index)

    def failed(self, node_index: int):
        self._notify(TASK_STATUS.FAILED, node_index)

    def read(self, node_index: int) -> TaskStatus:
        data = _best_effort_read_key(self._datastore, self._path(node_index))
        if not data:
            return TaskStatus(
                status=TASK_STATUS.UNREACHABLE, timestamp=None, node_index=node_index
            )
        data = json.loads(data.text)
        return TaskStatus(
            status=data["status"],
            timestamp=data["timestamp"],
            node_index=data["node_index"],
        )


class TaskUnreachableException(Exception):
    pass


class TaskFailedException(Exception):
    pass


class HeartbeatTimeoutException(Exception):
    pass


def wait_for_task_to_be_reachable(
    task_status_notifier: TaskStatusNotifier, node_index: int, timeout: int
):
    while task_status_notifier.read(node_index).status == TASK_STATUS.UNREACHABLE:
        time.sleep(1)
        timeout -= 1
        if timeout <= 0:
            # TODO: Improve timeout error messages to make it clearer to the user that
            # the task was unreachable and we hit a timeout for that.
            raise TimeoutError

    debug_logger(f"Task {node_index} is reachable")


def wait_for_task_completion(
    task_status_notifier: TaskStatusNotifier,
    node_index: int,
    heartbeat_timeout: int = 60 * 60,
    unreachable_timeout: int = 60 * 5,
):
    wait_for_task_to_be_reachable(task_status_notifier, node_index, unreachable_timeout)

    task_unreachable_since = None
    while True:
        status = task_status_notifier.read(node_index)
        debug_logger(
            f"Task {node_index} status: {status.status} with timestamp {status.timestamp}"
        )

        if status.status == TASK_STATUS.RUNNING:
            task_unreachable_since = None

        if status.status == TASK_STATUS.FINISHED:  # Success path
            debug_logger(f"Task {node_index} finished")
            return
        if status.status == TASK_STATUS.FAILED:  # Failure path
            debug_logger(f"Task {node_index} failed")
            raise TaskFailedException

        if status.status == TASK_STATUS.UNREACHABLE:
            debug_logger(f"Task {node_index} unreachable")
            if task_unreachable_since is None:
                task_unreachable_since = time.time()

            if time.time() - task_unreachable_since > unreachable_timeout:
                debug_logger(f"Task Reached {node_index} unreachable timeout")
                raise TaskUnreachableException
            else:
                debug_logger(
                    f"Task {node_index} still unreachable after {round(time.time() - task_unreachable_since, 2)} seconds"
                )

        elif time.time() - status.timestamp > heartbeat_timeout:  # Failure path
            debug_logger(f"Task {node_index} heartbeat timeout")
            raise HeartbeatTimeoutException

        time.sleep(1)


class HeartbeatThread(Thread):
    def __init__(
        self,
        task_status_notifier: TaskStatusNotifier,
        node_index: int,
        heartbeat_interval: int = 10,
    ):
        super().__init__()
        self.task_status_notifier = task_status_notifier
        self.node_index = node_index
        self.heartbeat_interval = heartbeat_interval
        self._exit_event = Event()
        self._running = False

    def run(self):
        self._running = True
        while self._exit_event.is_set() is False:
            self.task_status_notifier.heartbeat(self.node_index)
            time.sleep(self.heartbeat_interval)

    def stop(self):
        if self._running:
            self._exit_event.set()
            self.join()
