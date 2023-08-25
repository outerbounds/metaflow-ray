from metaflow import (
    FlowSpec,
    step,
    ray_parallel,
    batch,
    card,
    current,
    environment,
    pip_base,
)
from decorators import gpu_profile
from metaflow.metaflow_config import DATATOOLS_S3ROOT

NUM_NODES = 2
DATA_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(cpu=4, gpu=1, memory=32000, use_tmpfs=True, tmpfs_size=8000)
DEPS = dict(
    packages={
        "ray": "2.6.3",
        "xgboost": "",
        "xgboost_ray": "",
        "s3fs": "",
        "matplotlib": "",
        "pyarrow": "",
    },
)


@pip_base(**DEPS)
class RayXGBoostMultinodeGPU(FlowSpec):

    n_files = 1500
    n_cpu = RESOURCES["cpu"]
    n_gpu = RESOURCES["gpu"]
    s3_url = DATA_URL

    @step
    def start(self):
        self.next(self.train, num_parallel=NUM_NODES)

    @environment(
        vars={
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
        }
    )
    @ray_parallel
    @gpu_profile(interval=1)
    @batch(**RESOURCES)
    @card
    @step
    def train(self):

        import os
        import ray
        from metaflow import S3
        from table_loader import load_table
        from xgb_example import load_data, fit_model

        # Initialize ray driver on the cluster @ray_parallel created.
        ray.init()

        # Load many files from S3 using Metaflow + PyArrow.
        # Then convert to Ray.dataset.
        table = load_table(self.s3_url, self.n_files, drop_cols=["row_id"])
        train_dataset, valid_dataset = load_data(table=table)

        self.result = fit_model(
            train_dataset,
            valid_dataset,
            n_cpu=self.n_cpu,
            n_gpu=self.n_gpu,
            num_workers=NUM_NODES,
            objective="reg:squarederror",
            eval_metric=["rmse"],
            run_config_storage_path=self.checkpoint_path,
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @step
    def end(self):
        print(self.result)


if __name__ == "__main__":
    RayXGBoostMultinodeGPU()
