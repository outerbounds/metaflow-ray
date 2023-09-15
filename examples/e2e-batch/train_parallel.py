from metaflow import FlowSpec, step, current, ray_parallel, batch, pypi
from base import TabularBatchPrediction

N_NODES = 8
COMMON_PKGS = {
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


class Train(FlowSpec, TabularBatchPrediction):
    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @batch
    @ray_parallel
    @pypi(packages=COMMON_PKGS)
    @step
    def train(self):
        from ray.air import RunConfig

        self.setup(n_nodes=N_NODES)
        non_default_trainer_args = dict(
            # self.checkpoint_path is automatically set by the @ray_parallel decorator
            run_config=RunConfig(storage_path=self.checkpoint_path)
        )
        self.result = self.load_trainer(non_default_trainer_args).fit()
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
            result = run.data.result
        """
        )


if __name__ == "__main__":
    Train()
