from metaflow import FlowSpec, step, metaflow_ray, current, kubernetes, pypi, Parameter

NUM_NODES = 10
N_CPU = 8
MEMORY = 12228
COMMON_PKGS = {
    "ray[train]": "2.6.3",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


class RayRemoteExample(FlowSpec):

    max_time_to_run = Parameter(
        "max-time-to-run", help="Max time to run the flow", default=60 * 6, type=int
    )

    def _do_ray_job(self):
        import ray
        import time

        ray.init()
        print("Ray initialized in the %s step." % current.step_name)
        print("Ray nodes: ", ray.nodes())
        print("Ray cluster resources:")
        for k, v in ray.cluster_resources().items():
            if "memory" in k.lower():
                print("%s: %sGB" % (k, round(int(v) / (1024 * 1024 * 1024), 2)))
            else:
                print("%s: %s" % (k, v))

        from dataframe_process import do_something

        start_time = time.time()
        while time.time() - start_time < self.max_time_to_run:
            do_something(num_chunks=NUM_NODES)

    @step
    def start(self):
        self.next(self.big_step, num_parallel=NUM_NODES)

    @pypi(packages=COMMON_PKGS)
    @kubernetes
    @metaflow_ray
    @step
    def big_step(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayRemoteExample()
