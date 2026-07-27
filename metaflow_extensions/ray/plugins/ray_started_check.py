# The purpose of this file is to ensure that we call `ray.init` outside the metaflow step process and in
# turn extract all the nodes that are a part of the ray cluster without having to mess up the user's
# runtime environment by calling `ray.init` inside the metaflow step process.
import json
import sys


def check_ray_started(head_address):
    import ray

    # Connect to the cluster explicitly instead of letting `ray.init()` discover it.
    # The decorator starts ray with a custom `--temp-dir`, so the address file that
    # `ray.init()` reads by default (/tmp/ray/ray_current_cluster) is never written,
    # and an address-less `ray.init()` would silently start its own 1-node cluster.
    ray.init(address=head_address)
    ray_nodes = ray.nodes()
    print(json.dumps(ray_nodes))


if __name__ == "__main__":
    check_ray_started(sys.argv[1])
