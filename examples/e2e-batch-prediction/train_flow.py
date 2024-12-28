from gpu_profile import gpu_profile
from base import TabularBatchPrediction
from metaflow import FlowSpec, step, kubernetes, metaflow_ray, pypi, current


class Train(FlowSpec, TabularBatchPrediction):

    def _do_ray_job(self):
        from ray.air import RunConfig

        self.setup(n_cpu=8, n_gpu=1, n_nodes=2)
        trainer_args = dict(
            run_config=RunConfig(
                storage_path=current.ray_storage_path
            )
        )
        trainer = self.load_trainer(trainer_args)
        self.result = trainer.fit()

    @step
    def start(self):
        self.next(self.train, num_parallel=2)

    @gpu_profile(interval=1)
    @kubernetes(memory=12228, cpu=8, gpu=1)
    @metaflow_ray
    @pypi(
        python="3.10",
        packages={
            "ray[air]": "2.40.0",
            "xgboost": "2.1.3",
            "xgboost-ray": "0.1.19",
            "pyarrow": "18.1.0",
            "tqdm": "4.67.1",
            "matplotlib": "3.10.0",
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
