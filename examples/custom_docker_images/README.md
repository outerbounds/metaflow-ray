# Introduction

The following three files showcase how to use custom images on remote execution environments such as `@kubernetes` along with using the `@metaflow_ray` decorator.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow_with_gpu_image.py`

2. `flow_with_cpu_image.py` contains a flow that uses a custom image i.e. `registry.hub.docker.com/rayproject/ray:latest` with `@kubernetes`. Since this image already has `ray` installed, we don't need `@pypi(packages={"ray": "2.40.0"})`.

- This can be run using: `python examples/custom_docker_images/flow_with_cpu_image.py run`

3. `flow_with_gpu_image.py` contains a flow that uses a custom image i.e. `registry.hub.docker.com/rayproject/ray:latest-gpu` with `@kubernetes`. This image already has `ray` installed, but the `@gpu_profile` decorator also needs `matplotlib` to draw plots. Thus, `@pypi(packages={"matplotlib": "3.9.3"})` is also used.

- This can be run using: `python examples/custom_docker_images/flow_with_gpu_image.py --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/custom_docker_images/flow_with_gpu_image.py --environment=fast-bakery run`
