from metaflow import FlowSpec, step, pypi, kubernetes, environment, current, metaflow_ray
from gpu_profile import gpu_profile


class RayXGBoostGPU(FlowSpec):
    n_files = 1500
    data_url = "s3://obp-475b0e-metaflow/metaflow/mf.datasets/investment_ids"

    def _do_ray_job(self):
        import ray
        from utils import load_table, load_data, fit_model

        ray.init()

        table = load_table(self.data_url, self.n_files, drop_cols=["row_id"])
        train_dataset, valid_dataset = load_data(table=table)

        self.result = fit_model(
            train_dataset,
            valid_dataset,
            n_cpu=8,
            n_gpu=1,
            num_workers=2,
            objective="reg:squarederror",
            eval_metric=["rmse"],
            run_config_storage_path=current.ray_storage_path,
        )

    @step
    def start(self):
        self.next(self.train, num_parallel=2)

    @gpu_profile(interval=1)
    @kubernetes(cpu=8, gpu=1, memory=16000, use_tmpfs=True, tmpfs_size=4000)
    @metaflow_ray
    @pypi(packages={
        "ray[train]": "2.40.0",
        "xgboost": "2.1.3",
        "xgboost-ray": "0.1.19",
        "pyarrow": "18.1.0",
        "tqdm": "4.67.1",
        "matplotlib": "3.9.3",
    })
    @environment(
        vars={
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
        }
    )
    @step
    def train(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @pypi(packages={"ray[train]": "2.40.0"})
    @step
    def end(self):
        print(self.result)


if __name__ == "__main__":
    RayXGBoostGPU()
