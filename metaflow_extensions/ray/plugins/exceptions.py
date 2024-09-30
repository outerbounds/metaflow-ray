from metaflow.exception import MetaflowException


class RayNotInstalledException(MetaflowException):
    headline = "`ray` not installed"

    def __init__(self):
        msg = "ray is not installed. Please install ray before using the @metaflow_ray decorator."
        super(RayNotInstalledException, self).__init__(msg)


class DatastoreKeyNotFoundError(MetaflowException):
    headline = "Key not found"

    def __init__(self, datastore_path_name):
        msg = "Datastore path {} was not found.".format(datastore_path_name)
        super(DatastoreKeyNotFoundError, self).__init__(msg)


class BarrierTimeoutException(MetaflowException):
    headline = "Barrier Timeout"

    def __init__(self, lock_name, description):
        msg = f"Task has timed out after waiting for some keys to be written to the datastore.\n[Barrier Name]:{lock_name}\n[Barrier Info]: {description}"
        super(BarrierTimeoutException, self).__init__(msg)


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @metaflow_ray(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)


class ControlNodeHostNotReachableException(MetaflowException):
    headline = "Control node host error"

    def __init__(self, host, task_id):
        msg = (
            "The control node host (%s)[task-id: %s] is not reachable. Please check the host is reachable."
            % (host, task_id)
        )
        super(ControlNodeHostNotReachableException, self).__init__(msg)


class ControlTaskException(MetaflowException):
    headline = "Contral task error"

    def __init__(self, e):
        msg = """
Spinning down all workers because of the following exception running the @step code on the control task:
    {exception_str}
        """.format(
            exception_str=str(e)
        )
        super(ControlTaskException, self).__init__(msg)


class RayException(MetaflowException):
    headline = "Metaflow Ray Exception"

    def __init__(self, msg):
        super(RayException, self).__init__(msg)


class RayWorkerFailedStartException(MetaflowException):
    headline = "Worker task startup error"

    def __init__(self, node_index):
        msg = "Worker task failed to start on node {}".format(node_index)
        super(RayWorkerFailedStartException, self).__init__(msg)


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @metaflow_ray(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)
