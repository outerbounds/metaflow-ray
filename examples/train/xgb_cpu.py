from metaflow import FlowSpec, step, batch, Parameter, S3, current, pypi
from metaflow.metaflow_config import DATATOOLS_S3ROOT

COMMON_PKGS = {
    "ray[train]": "2.7.1",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}
DATA_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=32000, cpu=8, use_tmpfs=True, tmpfs_size=8000)


class RayXGBoostCPU(FlowSpec):
    n_files = 500
    n_cpu = RESOURCES["cpu"]
    s3_url = DATA_URL

    @batch
    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.train)

    @pypi(packages=COMMON_PKGS)
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
        )

        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @batch
    @step
    def end(self):
        print(self.result.metrics)


if __name__ == "__main__":
    RayXGBoostCPU()
