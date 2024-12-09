from metaflow import FlowSpec, step, pypi, kubernetes, metaflow_ray


class RayDFProcessFlow(FlowSpec):

    def _do_ray_job(self):
        import ray
        import numpy as np
        import pandas as pd
        from utils import process_dataframe

        ray.init()

        sample_df = pd.DataFrame({"existing_column": np.random.randint(1, 100, size=1000000)})
        custom_function = lambda x: x**2
        processed_df = process_dataframe(sample_df, custom_function, num_chunks=10)
        print(processed_df.head())

    @step
    def start(self):
        self.next(self.execute, num_parallel=10)

    @kubernetes
    @metaflow_ray
    @pypi(python="3.10", packages={"ray": "2.39.0", "pandas": "2.2.3", "numpy": "2.2.0"})
    @step
    def execute(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayDFProcessFlow()
