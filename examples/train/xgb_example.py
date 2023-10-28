import ray

from metaflow import profile
from concurrent.futures import ThreadPoolExecutor

import logging
import warnings

from pyarrow import Table


def load_data(
    s3_uri: str = None, table: Table = None, split_ratio: float = 0.2, n_cpu: int = 8
):
    if s3_uri is not None and table is not None:
        raise ValueError(
            "load_data only supports s3:// paths or a pyarrow.Table, not both."
        )
    if s3_uri is None and table is None:
        raise ValueError("load_data requires either a s3:// path or a pyarrow.Table.")
    if s3_uri is not None and s3_uri.endswith(".csv"):
        dataset = ray.data.read_csv(s3_uri)
    elif s3_uri is not None and s3_uri.endswith(".parquet"):
        dataset = ray.data.read_parquet(s3_uri)
    elif table is not None and isinstance(table, Table):
        dataset = ray.data.from_arrow(table)
    else:
        raise ValueError(
            "load_data only supports .csv, .parquet, or a directory of objects."
        )
    train_dataset, valid_dataset = dataset.train_test_split(test_size=split_ratio)
    logging.log(
        logging.INFO,
        f"Loaded {train_dataset.count()} training rows and {valid_dataset.count()} validation rows.",
    )
    return train_dataset, valid_dataset


def fit_model(
    train_dataset: ray.data.dataset.Dataset,
    valid_dataset: ray.data.dataset.Dataset,
    num_boost_round: int = 10,
    num_workers: int = 1,
    n_cpu: int = 4,
    n_gpu: int = 0,
    objective="reg:squarederror",
    eval_metric=["rmse"],
    run_config_storage_path=None,
):
    from xgboost_ray import RayParams
    from ray.train.xgboost import XGBoostTrainer
    from ray.air.config import ScalingConfig
    from ray.air import RunConfig

    if run_config_storage_path is not None:
        run_config = RunConfig(storage_path=run_config_storage_path)
    else:
        run_config = None

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=n_gpu > 0,
        resources_per_worker={"CPU": n_cpu, "GPU": n_gpu}
    )
    trainer = XGBoostTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
        label_column="target",
        num_boost_round=num_boost_round,
        params={
            "objective": objective,
            "tree_method": "gpu_hist" if n_gpu > 0 else "hist",
            "eval_metric": eval_metric,
        },
        datasets={"train": train_dataset, "valid": valid_dataset},
    )
    result = trainer.fit()
    return result


if __name__ == "__main__":
    ray.init()
    train_dataset, valid_dataset = load_data(
        "s3://anonymous@air-example-data/breast_cancer.csv"
    )
    result = fit_model(train_dataset, valid_dataset)
    print(result.metrics)
