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


def ensure_ray_installed():
    try:
        import ray
    except ImportError:
        raise RayNotInstalledException


def warning_message(message, prefix="[@metaflow_ray]"):
    msg = "%s %s" % (prefix, message)
    print(msg, file=sys.stderr)


def start_ray_processes(ubf_context, main_ip, main_port, node_index):
    # When ray processes start and finish properly it means that the process
    # would have successfully registered as a part of the cluster.
    import ray

    try:
        if ubf_context == UBF_CONTROL:
            runtime_start_result = subprocess.run(
                [
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
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        else:
            node_ip_address = ray._private.services.get_node_ip_address()
            runtime_start_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ray.scripts.scripts",
                    "start",
                    "--node-ip-address",
                    node_ip_address,
                    "--address",
                    "%s:%s" % (main_ip, main_port),
                    "--disable-usage-stats",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        process_type = "control" if ubf_context == UBF_CONTROL else "worker"
        e.stderr = e.stderr.replace("\n", "\n\t")
        e.stdout = e.stdout.replace("\n", "\n\t")
        raise RayException(
            "Ray processes [%s][on node-index %s] failed to start with exception:\n%s\n%s"
            % (process_type, str(node_index), e.stderr, e.stdout)
        )
    warning_message(
        "Ray processes started successfully on node-index %s [%s]"
        % (
            str(node_index),
            "control" if ubf_context == UBF_CONTROL else "worker",
        )
    )
    return runtime_start_result


def _extract_ray_nodes():
    try:
        completed_proc = subprocess.run(
            [sys.executable, RAY_NODE_EXTRACTOR_FILE, resolve_main_ip()],
            check=True,
            capture_output=True,
        )
        data_str = completed_proc.stdout.decode()
        return json.loads(data_str)
    except subprocess.CalledProcessError as e:
        return None
    except json.JSONDecodeError:
        return None


def wait_for_ray_nodes_to_join(max_wait_time):
    # This function will wait untill all ray nodes have joined the cluster.
    # If nodes have not joined after a certain amount of timeout it will raise an exception.
    # We leverage subprocesses to extract the number of nodes that have joined the cluster.
    # We do this so that users don't face any error when they call `ray.init` in their user code.
    # Extracting number of nodes in a separate subprocess ensures that when users call `ray.init`,
    # ray will not end up throwing and exception.

    start_time = time.time()
    _iters = 0
    while True:
        ray_nodes = _extract_ray_nodes()
        if ray_nodes is not None:
            if len(ray_nodes) == current.parallel.num_nodes:
                warning_message(
                    "All `ray` nodes joined the cluster. Number of nodes in cluster: %s"
                    % str(len(ray_nodes))
                )
                return ray_nodes
        if _iters % 10 == 0:
            warning_message(
                "Waiting for all `ray` nodes to join the cluster. Current number of nodes in cluster: %s"
                % str(len(ray_nodes))
            )
        _iters += 1
        time.sleep(1)
        if time.time() - start_time > max_wait_time:
            raise RayException(
                "All `ray` nodes did not join the cluster in %s seconds."
                % max_wait_time
            )
