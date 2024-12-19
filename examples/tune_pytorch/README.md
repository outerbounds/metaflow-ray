# Introduction

The following four files showcase how to tune PyTorch models using the `@metaflow_ray` decorator with `@kubernetes`.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow_gpu.py`

2. `utils.py` contains helper functions to train and test a custom PyTorch model.

3. `flow_cpu.py` contains a flow that uses `@metaflow_ray` with `@kubernetes` to tune the PyTorch model.

- This can be run using: `python examples/tune_pytorch/flow_cpu.py --no-pylint --environment=pypi run`

4. `flow_gpu.py` contains a flow that uses `@metaflow_ray` with `@kubernetes` to tune the PyTorch model. It also passes in `gpu` requirement to `@kubernetes` and the `ScalingConfig` when defining the search space.

- This can be run using: `python examples/tune_pytorch/flow_gpu.py --no-pylint --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/tune_pytorch/flow_gpu.py --no-pylint --environment=fast-bakery run`
