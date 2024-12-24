# Introduction

The following four files showcase how to tune models using the `@metaflow_ray` decorator with `@kubernetes` with the PyTorch Lightning library.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow.py`

2. `dataloader.py` contains utilities to load the MNIST dataset. It uses `LightningDataModule` under the hood.

3. `model.py` contains a simple neural network for classifying MNIST images. It uses `LightningModule` under the hood.

4. `flow.py` contains a flow that uses `@metaflow_ray` with `@kubernetes` to tune the PyTorch Lightning model. It also passes in `gpu` requirement to `@kubernetes` and the `ScalingConfig` when defining the `TorchTrainer`.

- This can be run using: `python examples/ray_torch_lightning/flow.py --no-pylint --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/ray_torch_lightning/flow.py --no-pylint --environment=fast-bakery run`
