from metaflow.exception import MetaflowException


class RayNotInstalledException(MetaflowException):
    headline = "[@metaflow_ray] The ray package is not installed in the decorator runtime"

    def __init__(self, step_name):
        msg = f"""You have these options:\n\t- In the {step_name} @step annotated with @metaflow_ray, add the latest pypi or conda version of ray to your @pypi or @conda decorator.\n\t- In the {step_name} @step annotated with @metaflow_ray, use a base Docker image with ray installed in your @kubernetes(image=<image>) decorator or --with kubenetes:image=<image> in the cli.
        """
        super(RayNotInstalledException, self).__init__(msg)


class DatastoreKeyNotFoundError(MetaflowException):
    headline = "[@metaflow_ray] Key not found"

    def __init__(self, datastore_path_name):
        msg = "Datastore path {} was not found.".format(datastore_path_name)
        super(DatastoreKeyNotFoundError, self).__init__(msg)


class BarrierTimeoutException(MetaflowException):
    headline = "[@metaflow_ray] Barrier timeout"

    def __init__(self, lock_name, description):
        msg = f"Task has timed out after waiting for some keys to be written to the datastore.\n[Barrier Name]:{lock_name}\n[Barrier Info]: {description}"
        super(BarrierTimeoutException, self).__init__(msg)


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "[@metaflow_ray] All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @metaflow_ray(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)


class ControlNodeHostNotReachableException(MetaflowException):
    headline = "[@metaflow_ray] Control node host error"

    def __init__(self, host, task_id):
        msg = (
            "The control node host (%s)[task-id: %s] is not reachable. Please check the host is reachable."
            % (host, task_id)
        )
        super(ControlNodeHostNotReachableException, self).__init__(msg)


class ControlTaskException(MetaflowException):
    headline = "[@metaflow_ray] Contral task error"

    def __init__(self, e):
        msg = """Spinning down all workers because of the following exception running the @step code on the control task:
    {exception_str}
        """.format(
            exception_str=str(e)
        )
        super(ControlTaskException, self).__init__(msg)


class RayException(MetaflowException):
    headline = "[@metaflow_ray] Metaflow Ray Exception"

    def __init__(self, msg):
        super(RayException, self).__init__(msg)


class RayWorkerFailedStartException(MetaflowException):
    headline = "[@metaflow_ray] Worker task startup error"

    def __init__(self, node_index):
        msg = "Worker task failed to start on node {}".format(node_index)
        super(RayWorkerFailedStartException, self).__init__(msg)


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "[@metaflow_ray] All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @metaflow_ray(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)
