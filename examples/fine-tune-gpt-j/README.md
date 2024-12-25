# Introduction

The following four files showcase how to finetune GPT-J model using the `@metaflow_ray` decorator with `@kubernetes`.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow.py`

2. `dataloader.py` contains helper functions to split text and tokenize it.

3. `trainer.py` contains utilities to train the GPT-J model using the `transformers` library.

4. `flow.py` uses `@metaflow_ray` with `@kubernetes` to finetune the GPT-J model. It also passes in `gpu` requirement to `@kubernetes` and the `ScalingConfig` of the `TorchTrainer`.

- The flow can be run using `python flow.py --no-pylint --environment=fast-bakery run`. This leverages `fast-bakery` for blazingly fast docker image builds on the [Outerbounds](https://outerbounds.com/) platform.
