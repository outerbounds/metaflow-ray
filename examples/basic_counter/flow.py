from metaflow import FlowSpec, step, pypi, kubernetes, metaflow_ray


class RayCounterFlow(FlowSpec):

    def _do_ray_job(self):
        import ray
        import time
        from counter import Counter

        ray.init()

        memory = ray.cluster_resources().get("memory")
        print("memory: %sGB" % (round(int(memory) / (1024 * 1024 * 1024), 2)))

        c = Counter.remote()
        for _ in range(10):
            time.sleep(1)
            c.incr.remote(1)

        print(ray.get(c.get.remote()))

    @step
    def start(self):
        self.next(self.execute, num_parallel=2)

    @kubernetes
    @metaflow_ray
    @pypi(packages={"ray": "2.39.0"})
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
    RayCounterFlow()
