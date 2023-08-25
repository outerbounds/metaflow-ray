from metaflow import FlowSpec, step, current, ray_parallel, batch, pip_base
from base import TabularBatchPrediction

N_NODES = 8

@pip_base(packages={"ray": "2.6.3", "pandas": "", "xgboost": "", "xgboost_ray": "", "pyarrow": ""})
class Train(FlowSpec, TabularBatchPrediction):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @batch
    @ray_parallel
    @step
    def train(self):
        from ray.air import RunConfig
        self.setup(n_nodes=N_NODES)
        non_default_trainer_args = dict(
            # self.checkpoint_path is automatically set by the @ray_parallel decorator
            run_config = RunConfig(storage_path=self.checkpoint_path)
        )
        self.result = self.load_trainer(non_default_trainer_args).fit()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

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
