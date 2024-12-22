import ray
from pyarrow import Table
from metaflow import S3, profile
from concurrent.futures import ThreadPoolExecutor


def _print_throughput(caption, stats, total_size):
    gbps = (total_size * 8) / (next(iter(stats.values())) / 1000.0)
    print("%s: %2.1f Gbit/s" % (caption, gbps))


def _load_s3(url, s3, num_files):
    files = list(s3.list_recursive([url]))[:num_files]
    total_size = sum(f.size for f in files) / 1024**3
    print("Loading %2.1dGB of data" % total_size)
    stats = {}

    with profile("download", stats_dict=stats):
        loaded = s3.get_many([f.url for f in files])
    _print_throughput("S3->EC2 download", stats, total_size)

    return [f.path for f in loaded], total_size


def load_table(
    url,
    num_files,
    num_threads=8,
    only_download=False,
    reduce_precision=True,
    drop_cols=["row_id"],
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    stats = {}
    with profile("read", stats_dict=stats):
        with S3() as s3:
            fls, total_size = _load_s3(url, s3, num_files)
            if only_download:
                return
            with ThreadPoolExecutor(max_workers=num_threads) as exe:
                tables = exe.map(lambda f: pq.read_table(f, use_threads=False), fls)
                table = pa.concat_tables(tables)
    
    _print_throughput(
        "Decoding throughput from S3 to PyArrow tables", stats, total_size
    )

    if reduce_precision:
        for i, (col_name, type_) in enumerate(
            zip(table.schema.names, table.schema.types)
        ):
            if pa.types.is_decimal(type_):
                table = table.set_column(
                    i, col_name, pa.cast(table.column(col_name), pa.float32())
                )

    if len(drop_cols) > 0:
        table = table.drop(drop_cols)

    return table


def load_data(
    s3_uri: str = None, table: Table = None, split_ratio: float = 0.2
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
    print(f"Loaded {train_dataset.count()} training rows and {valid_dataset.count()} validation rows.")

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
        resources_per_worker={"CPU": n_cpu, "GPU": n_gpu},
    )

    trainer = XGBoostTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
        label_column="target",
        num_boost_round=num_boost_round,
        params={
            "objective": objective,
            "tree_method": "hist",
            "eval_metric": eval_metric,
            "device": "cuda" if n_gpu > 0 else "cpu"
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
