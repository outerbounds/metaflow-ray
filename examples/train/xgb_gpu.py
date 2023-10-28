from metaflow import FlowSpec, step, batch, pypi, current, card, environment
from decorators import gpu_profile
from metaflow.metaflow_config import DATATOOLS_S3ROOT

DATA_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=16000, cpu=4, gpu=1, use_tmpfs=True, tmpfs_size=4000)
COMMON_PKGS = {
    "ray[train]": "2.7.1",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


class RayXGBoostGPU(FlowSpec):
    n_files = 500
    n_cpu = RESOURCES["cpu"]
    n_gpu = RESOURCES["gpu"]
    s3_url = DATA_URL

    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.train)

    @pypi(packages=COMMON_PKGS)
    @environment(
        vars={
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "NCCL_SOCKET_IFNAME": "eth0",
        }
    )
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

        # Initialize ray driver on the cluster @metaflow_ray created.
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
            objective="reg:squarederror",
            eval_metric=["rmse"],
        )

        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        print(self.result.path)


if __name__ == "__main__":
    RayXGBoostGPU()
