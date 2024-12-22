# Introduction

The following four files showcase how to train XGBoost models using the `@metaflow_ray` decorator with `@kubernetes`.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow_gpu.py`

2. `utils.py` contains helper functions to load dataset from S3 and to train an XGBoost model.

3. `flow_cpu.py` contains a flow that uses `@metaflow_ray` with `@kubernetes` to train the XGBoost model. It accepts the dataset through the `--data_url` parameter.

- This can be run using: `python examples/train_xgboost/flow_cpu.py --environment=pypi run --data_url "s3://obp-475b0e-metaflow/metaflow/mf.datasets/investment_ids"`

4. `flow_gpu.py` contains a flow that uses `@metaflow_ray` with `@kubernetes` to train the XGBoost model. It also passes in `gpu` requirement to `@kubernetes` and the `fit_model` function from `utils.py`. It accepts the dataset through the `--data_url` parameter.

- This can be run using: `python examples/train_xgboost/flow_gpu.py --environment=pypi run --data_url "s3://obp-475b0e-metaflow/metaflow/mf.datasets/investment_ids"`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/train_xgboost/flow_gpu.py --environment=fast-bakery run --data_url "s3://obp-475b0e-metaflow/metaflow/mf.datasets/investment_ids"`
