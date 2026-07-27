import os
import subprocess
import sys
import json
import time
from .exceptions import (
    RayException,
    ControlNodeHostNotReachableException,
    RayNotInstalledException,
)
from metaflow.metaflow_current import current
from metaflow.unbounded_foreach import UBF_CONTROL
from .constants import DASHBOARD_HOST, DEFAULT_DASHBOARD_PORT

RAY_NODE_EXTRACTOR_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ray_started_check.py"
)


def resolve_main_ip():
    main_ip = current.parallel.main_ip
    import socket

    try:
        return socket.gethostbyname(main_ip)
    except socket.gaierror:
        raise ControlNodeHostNotReachableException


def ensure_ray_installed(step_name):
    try:
        import ray
    except ImportError:
        raise RayNotInstalledException(step_name)


def warning_message(message, prefix="[@metaflow_ray]"):
    msg = "%s %s" % (prefix, message)
    print(msg, file=sys.stderr)


# Ray renamed this helper between versions, so try both before falling back to the
# layout ray documents (`<system temp dir>/ray`).
_TEMP_DIR_GETTERS = [
    ("ray._common.utils", "get_default_ray_temp_dir"),  # ray >= 2.40
    ("ray._private.utils", "get_ray_temp_dir"),  # older ray
]


def default_ray_temp_dir():
    # The root temp dir that `ray start` uses when `--temp-dir` is not passed. Ray keeps
    # the cluster address file and the session logs under here.
    import importlib

    for module_name, fn_name in _TEMP_DIR_GETTERS:
        try:
            return getattr(importlib.import_module(module_name), fn_name)()
        except (ImportError, AttributeError):
            continue
    import tempfile

    return os.path.join(tempfile.gettempdir(), "ray")


def start_ray_processes(
    ubf_context,
    main_ip,
    main_port,
    node_index,
    logging_level=None,
    log_style=None,
    enable_dashboard=False,
    dashboard_port=DEFAULT_DASHBOARD_PORT,
):
    # When ray processes start and finish properly it means that the process
    # would have successfully registered as a part of the cluster.
    import ray

    try:
        if ubf_context == UBF_CONTROL:
            cmd = [
                sys.executable,
                "-m",
                "ray.scripts.scripts",
                "start",
                "--head",
                "--node-ip-address",
                main_ip,
                "--port",
                str(main_port),
                "--disable-usage-stats",
            ]
            if enable_dashboard:
                # The dashboard only ever runs on the head node, and `ray start`
                # rejects these flags when `--head` is absent.
                cmd.extend(
                    [
                        "--include-dashboard",
                        "true",
                        "--dashboard-host",
                        DASHBOARD_HOST,
                        "--dashboard-port",
                        str(dashboard_port),
                    ]
                )
            if logging_level:
                cmd.extend(["--logging-level", logging_level])
            if log_style:
                cmd.extend(["--log-style", log_style])

            runtime_start_result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

        else:
            node_ip_address = ray._private.services.get_node_ip_address()
            cmd = [
                sys.executable,
                "-m",
                "ray.scripts.scripts",
                "start",
                "--node-ip-address",
                node_ip_address,
                "--address",
                "%s:%s" % (main_ip, main_port),
                "--disable-usage-stats",
            ]
            if logging_level:
                cmd.extend(["--logging-level", logging_level])
            if log_style:
                cmd.extend(["--log-style", log_style])

            runtime_start_result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        process_type = "control" if ubf_context == UBF_CONTROL else "worker"
        e.stderr = e.stderr.replace("\n", "\n\t")
        e.stdout = e.stdout.replace("\n", "\n\t")
        message = (
            "Ray processes [%s][on node-index %s] failed to start with exception:\n%s\n%s"
            % (process_type, str(node_index), e.stderr, e.stdout)
        )
        if ubf_context == UBF_CONTROL and enable_dashboard:
            # `--include-dashboard true` makes ray start fail hard if the dashboard's
            # dependencies are missing, instead of warning and moving on.
            message += (
                "\n\t`@metaflow_ray(enable_dashboard=True)` requires the dashboard's dependencies "
                "to be installed in the execution environment (`pip install 'ray[default]'`). "
                "Either install them or set `enable_dashboard=False`."
            )
        raise RayException(message)
    warning_message(
        "Ray processes started successfully on node-index %s [%s]"
        % (
            str(node_index),
            "control" if ubf_context == UBF_CONTROL else "worker",
        )
    )
    if ubf_context == UBF_CONTROL and enable_dashboard:
        warning_message(
            "Ray dashboard is running on port %s of the control task [http://%s:%s]. "
            % (
                str(dashboard_port),
                main_ip,
                str(dashboard_port),
            )
        )
    return runtime_start_result


def _extract_ray_nodes(head_address):
    # Returns a (nodes, error) tuple. `nodes` is None when the node list could not be
    # read, in which case `error` explains why; swallowing that error makes cluster
    # formation problems impossible to diagnose from the task logs.
    try:
        completed_proc = subprocess.run(
            [sys.executable, RAY_NODE_EXTRACTOR_FILE, head_address],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return None, e.stderr.decode(errors="replace").strip()
    data_str = completed_proc.stdout.decode()
    try:
        return json.loads(data_str), None
    except json.JSONDecodeError:
        return None, "Could not parse the `ray` node list: %s" % data_str.strip()


def wait_for_ray_nodes_to_join(max_wait_time, main_ip, main_port):
    # This function will wait untill all ray nodes have joined the cluster.
    # If nodes have not joined after a certain amount of timeout it will raise an exception.
    # We leverage subprocesses to extract the number of nodes that have joined the cluster.
    # We do this so that users don't face any error when they call `ray.init` in their user code.
    # Extracting number of nodes in a separate subprocess ensures that when users call `ray.init`,
    # ray will not end up throwing and exception.

    head_address = "%s:%s" % (main_ip, main_port)
    start_time = time.time()
    _iters = 0
    while True:
        ray_nodes, probe_error = _extract_ray_nodes(head_address)
        if ray_nodes is not None:
            if len(ray_nodes) == current.parallel.num_nodes:
                warning_message(
                    "All ray nodes joined the cluster. Number of nodes in cluster: %s"
                    % str(len(ray_nodes))
                )
                time.sleep(1)
                return ray_nodes
        if _iters % 10 == 0:
            warning_message(
                "Waiting for all `ray` nodes to join the cluster. Current number of nodes in cluster: %s"
                % (str(len(ray_nodes)) if ray_nodes is not None else "unknown")
            )
            if probe_error:
                warning_message(
                    "Could not read the `ray` node list from %s: %s"
                    % (head_address, probe_error)
                )
        _iters += 1
        time.sleep(1)
        if time.time() - start_time > max_wait_time:
            message = (
                "All `ray` nodes did not join the cluster in %s seconds."
                % max_wait_time
            )
            if probe_error:
                message += (
                    " The last attempt to read the node list from %s failed with: %s"
                    % (head_address, probe_error)
                )
            raise RayException(message)
