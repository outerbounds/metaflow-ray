from metaflow import FlowSpec, step, batch, Parameter, S3, current, pip_base
from metaflow.metaflow_config import DATATOOLS_S3ROOT

DATA_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=32000, cpu=8, use_tmpfs=True, tmpfs_size=8000)
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
class RayXGBoostCPU(FlowSpec):

    n_files = 500
    n_cpu = RESOURCES["cpu"]
    s3_url = DATA_URL

    @step
    def start(self):
        self.next(self.train)

    @batch(**RESOURCES)
    @step
    def train(self):

        import os
        import ray
        from metaflow import S3
        from table_loader import load_table
        from xgb_example import load_data, fit_model

        ray.init()

        table = load_table(self.s3_url, self.n_files, drop_cols=["row_id"])
        train_dataset, valid_dataset = load_data(table=table)

        self.result = fit_model(
            train_dataset,
            valid_dataset,
            n_cpu=self.n_cpu,
            objective="reg:squarederror",
            eval_metric=["rmse"],
            run_config_storage_path=self.checkpoint_path,
        )

        self.next(self.end)

    @step
    def end(self):
        print(self.result.metrics)


if __name__ == "__main__":
    RayXGBoostCPU()
