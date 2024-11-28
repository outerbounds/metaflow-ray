import os
import sys
from functools import partial
import json
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
)
from metaflow.metaflow_current import current
from .status_notifier import (
    TaskStatusNotifier,
    HeartbeatThread,
    HeartbeatTimeoutException,
    TaskFailedException,
    wait_for_task_completion,
)
from .exceptions import ControlNodeHostNotReachableException, RayException
from .datastore import task_sync_barrier, DecoratorDatastore
from .ray_utils import (
    ensure_ray_installed,
    resolve_main_ip,
    start_ray_processes,
    warning_message,
    wait_for_ray_nodes_to_join,
)
from .constants import RAY_SUFFIX

def _worker_node_heartbeat_monitor(
    datastore: DecoratorDatastore, node_index: int, heartbeat_timeout=60 * 10
):
    """
    The worker tasks will poll for the control task's heartbeat and do nothing else.
    Any failure in the worker's entry-point script will result in the failure at the control task level because the worker won't join the cluster.
    We ensure that that all nodes are available before user code starts execution.

    Since we are not letting the user code execute in the worker node, We only need to ensure that control task is properly running
    since it holds the user code that will execute in the ray cluster.

    Since the decorator is only adding functionality on top of the compute orchestration decorators like k8s, batch etc, The "failure" of any
    container will be managed by the higher level compute orchestrator; Failure modes that are possible:
    - User code in control task fails
    - Control task fails intermittently.
    - The worker/control fails intermittently such as node being wiped off
    """
    # TODO : Make heartbeat timeout configurable
    _status_notifier = TaskStatusNotifier(datastore)
    # Worker task statuses are only for bookkeeping.
    # They are not used by the control task in any way.
    _status_notifier.running(node_index)
    try:
        # Poll the control task's heartbeat and fail if control task fails
        # or if the heartbeat interval crosses the threshold.
        wait_for_task_completion(
            _status_notifier, node_index=0, heartbeat_timeout=heartbeat_timeout
        )
        _status_notifier.finished(node_index)
    except HeartbeatTimeoutException:
        _status_notifier.failed(node_index)
        raise RayException(
            f"Control task heartbeat timed out. Control task has not published a heartbeat for {heartbeat_timeout} seconds."
        )
    except TaskFailedException:
        _status_notifier.failed(node_index)
        raise RayException("Control task reported failure.")


class RayDecorator(ParallelDecorator):
    name = "metaflow_ray"
    defaults = {
        "main_port": None,
        "worker_polling_freq": 10,  # We DONT use this anymore
        "all_nodes_started_timeout": 90,
    }
    IS_PARALLEL = True

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        super().step_init(
            flow, graph, step_name, decorators, environment, flow_datastore, logger
        )
        self.flow_datastore = flow_datastore
        self._heartbeat_thread = None
    
    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        super().task_pre_step(
            step_name,
            task_datastore,
            metadata,
            run_id,
            task_id,
            flow,
            graph,
            retry_count,
            max_user_code_retries,
            ubf_context,
            inputs,
        )
        self.ubf_context = ubf_context
        ensure_ray_installed(step_name)
        self.deco_datastore = DecoratorDatastore(
            self.flow_datastore,
            "%s/%s/%s/%s" % (flow.name, run_id, step_name, task_id),
            retry_count,
        )

        storage_root = self.deco_datastore.get_storage_root
        if storage_root.startswith(RAY_SUFFIX):
            from metaflow.metaflow_config import DATASTORE_SYSROOT_S3
            
            storage_root = os.path.join(DATASTORE_SYSROOT_S3, storage_root)

        current._update_env({
            "ray_storage_path": os.path.join(storage_root, "%s/%s/%s" % (flow.name, run_id, step_name))
        })

    def _resolve_port(self):
        main_port = self.attributes["main_port"]
        if main_port is None:
            return 6379

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):

        # Since worker tasks are all monitoring the control task's heartbeat,
        # any exception in the control task will result in the worker tasks failing.
        if self.ubf_context == UBF_CONTROL:
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.stop()
                self._heartbeat_thread.task_status_notifier.failed(0)

    def wait_for_all_nodes_to_start(self):
        _control_key = "control_started.json"
        _worker_keys = [
            f"node_{i}_started.json" for i in range(1, current.parallel.num_nodes)
        ]
        max_wait_time = self.attributes["all_nodes_started_timeout"] or 300  #
        if self.ubf_context == UBF_CONTROL:
            self.deco_datastore.put(
                _control_key, json.dumps({"started": True}), overwrite=True
            )
        else:
            self.deco_datastore.put(
                f"node_{current.parallel.node_index}_started.json",
                json.dumps({"started": True}),
                overwrite=True,
            )
        task_sync_barrier(
            barrier_name="@metaflow_ray(node-index=%s)"
            % str(current.parallel.node_index),
            datastore=self.deco_datastore,
            keys=[_control_key] + _worker_keys,
            max_wait_time=max_wait_time,
            description=(
                "Job crashed because all workers didnot end up starting in %s seconds."
                "Increase the `all_nodes_started_timeout` in @metaflow_ray decorator to wait for longer."
                % max_wait_time
            ),
            wait_message="Waiting for all workers to start up",
        )

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        local_mode_control_task = (
            ubf_context == UBF_CONTROL
            and os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"
        )

        def control_task_function(
            status_notifier: TaskStatusNotifier, heartbeat_thread: HeartbeatThread
        ):
            # The control task will start a heartbeat thread that will publish heartbeats
            # These heartbeats will be monitored by the worker tasks.
            status_notifier.running(0)
            self.setup_distributed_env(flow)
            heartbeat_thread.start()
            try:
                step_func()
                status_notifier.finished(0)
            finally:
                warning_message(
                    "Stopping heartbeat thread for control task. Control task has finished."
                )
                heartbeat_thread.stop()

        def worker_task_function():
            # We first call self.setup_distributed_env so that all worker
            # nodes have the ray processes started and there is a barrier
            # that will ensure that user code execution will only start when
            # all nodes have started. This ensures that user code will have
            # access to a ray cluster with expected number of nodes.
            self.setup_distributed_env(flow)
            # The worker tasks will wait for the control task's heartbeat.
            # if it reaches a point where the control task failed for some reason
            # or the control task stopped publishing heartbeats, the worker task will fail.
            _worker_node_heartbeat_monitor(
                self.deco_datastore,
                current.parallel.node_index,
                heartbeat_timeout=10 * 60,  # 10 minutes (todo: make this configurable)
            )

        # A status notifier helps the control node publish heartbeats
        # and it helps the worker nodes monitor the control node's heartbeat.
        _status_notifier = TaskStatusNotifier(self.deco_datastore)
        # We only start the heartbeat "creation" thread for the control task.
        # This thread will publish heartbeats to the datastore from the control
        # task.
        self._heartbeat_thread = HeartbeatThread(
            _status_notifier, current.parallel.node_index, 5
        )
        __control_task_func = partial(
            control_task_function, _status_notifier, self._heartbeat_thread
        )
        if local_mode_control_task:
            # If it is a local mode control task then we need
            # to ensure we follow the same pattern as the parent decorator
            env_to_use = getattr(self.environment, "base_env", self.environment)
            return partial(
                _local_multinode_control_task_step_func,
                flow,
                env_to_use,
                __control_task_func,
                retry_count,
                ",".join(self.input_paths),  # self.input_paths set in parent class.
            )
        elif ubf_context == UBF_CONTROL:
            return __control_task_func
        else:
            return worker_task_function

    def setup_distributed_env(self, flow):
        """
        This function will setup the same Ray environment for all tasks (control and worker):
        - Wait for tasks to have started.
        - start the subsequent ray processes.
        - Wait for all ray nodes to join the cluster (on both worker and control.)
        """
        self.wait_for_all_nodes_to_start()
        main_port = self._resolve_port()
        main_ip = resolve_main_ip()
        start_ray_processes(
            self.ubf_context, main_ip, main_port, current.parallel.node_index
        )
        wait_for_ray_nodes_to_join(self.attributes["all_nodes_started_timeout"] or 300)
