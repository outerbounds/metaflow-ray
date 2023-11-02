from metaflow import FlowSpec, step, Parameter, current, batch, pypi, metaflow_ray
from base import TabularBatchPrediction

N_NODES = 2
COMMON_PKGS = {
    "ray[tune]": "2.6.3",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


class Tune(FlowSpec, TabularBatchPrediction):
    num_samples = Parameter(
        "n", help="Number of hyperparameter samples to run", default=10, type=int
    )

    optimization_metric = Parameter(
        "m", help="Metric to optimize", default="valid-logloss", type=str
    )

    optimization_direction = Parameter(
        "d", help="Direction to optimize", default="min", type=str
    )

    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.tune, num_parallel=N_NODES)

    @pypi(packages=COMMON_PKGS)
    @batch
    @metaflow_ray
    @step
    def tune(self):
        from metaflow.metaflow_config import DATATOOLS_S3ROOT
        from metaflow import current
        from ray.air.config import ScalingConfig
        from ray.air import RunConfig
        from ray import tune
        import os

        self.setup(n_nodes=N_NODES)

        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html#ray.tune.Tuner
        param_space = {
            "scaling_config": ScalingConfig(num_workers=1),
            "params": {
                "objective": "binary:logistic",
                "tree_method": "approx",
                "eval_metric": ["logloss", "error"],
                "eta": tune.loguniform(1e-4, 1e-1),
                "subsample": tune.uniform(0.5, 1.0),
                "max_depth": tune.randint(1, 9),
            },
        }

        # self.checkpoint_path is automatically set by the @metaflow_ray decorator
        run_config = RunConfig(storage_path=self.checkpoint_path)

        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
        tune_config = tune.TuneConfig(
            metric=self.optimization_metric,
            mode=self.optimization_direction,
            search_alg=tune.search.basic_variant.BasicVariantGenerator(),
            scheduler=tune.schedulers.ASHAScheduler(),
            num_samples=self.num_samples,
            time_budget_s=self.max_timeout,
            max_concurrent_trials=self.n_nodes,
        )
        tune_args = dict(
            trainable=self.load_trainer(),
            param_space=param_space,
            run_config=run_config,
            tune_config=tune_config,
        )
        tuner = self.load_tuner(tune_args)
        results = tuner.fit()
        self.results = results.get_dataframe()
        self.result = results.get_best_result()
        self.next(self.join)

    @pypi(packages=COMMON_PKGS)
    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        print(
            f"""

            Access result:

            from metaflow import Run
            run = Run('{current.flow_name}/{current.run_id}')
            df = run.data.results
            best_result = run.data.result
        """
        )


if __name__ == "__main__":
    Tune()
