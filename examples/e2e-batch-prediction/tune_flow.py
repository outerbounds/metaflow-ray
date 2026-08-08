from gpu_profile import gpu_profile
from base import TabularBatchPrediction
from metaflow import FlowSpec, step, kubernetes, metaflow_ray, pypi, current


class Tune(FlowSpec, TabularBatchPrediction):

    def _do_ray_job(self):
        from ray.air import RunConfig

        self.setup(n_cpu=8, n_gpu=1, n_nodes=2)
        tuner_args = dict(
            run_config=RunConfig(
                storage_path=current.ray_storage_path
            )
        )

        # run_config passed to tuner will be used by trainer also..
        tuner = self.load_tuner(trainer_args=None, tuner_args=tuner_args)
        results = tuner.fit()

        self.all_results = results.get_dataframe()
        self.best_result = results.get_best_result()

    @step
    def start(self):
        self.next(self.tune, num_parallel=2)

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
    def tune(self):
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
            all_results = run.data.all_results
            best_result = run.data.best_result
        """
        )


if __name__ == "__main__":
    Tune()
