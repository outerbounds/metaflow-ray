NODE_STARTED_VAR = "ray_node_started"
RAY_SUFFIX = "mf.ray_decorator"
DEFAULT_DASHBOARD_PORT = 8265
# Ray binds the dashboard to localhost by default, which makes it unreachable
# from outside the control task's container. We always bind to all interfaces.
DASHBOARD_HOST = "0.0.0.0"
