# The purpose of this file is to ensure that we call `ray.init` outside the metaflow step process and in
# turn extract all the nodes that are a part of the ray cluster without having to mess up the user's
# runtime environment by calling `ray.init` inside the metaflow step process.
import json
import sys


def check_ray_started(main_node_ip):
    import ray

    ray.init(
        _node_ip_address=main_node_ip,
    )
    ray_nodes = ray.nodes()
    print(json.dumps(ray_nodes))


if __name__ == "__main__":
    check_ray_started(sys.argv[1])
