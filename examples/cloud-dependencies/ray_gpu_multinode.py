from metaflow import FlowSpec, step, metaflow_ray, batch, current
from decorators import gpu_profile

NUM_NODES = 2
RESOURCES = {"gpu": 1, "memory": 12228}


class RayGPU(FlowSpec):
    def _do_ray_job(self):
        import ray

        ray.init()
        print("Ray initialized in the %s step." % current.step_name)
        for k, v in ray.cluster_resources().items():
            if "memory" in k.lower():
                print("%s: %sGB" % (k, round(int(v) / (1024 * 1024 * 1024), 2)))
            else:
                print("%s: %s" % (k, v))

    @step
    def start(self):
        self.next(self.big_step, num_parallel=NUM_NODES)

    @gpu_profile(interval=1)
    @batch(**RESOURCES, image="rayproject/ray:nightly-gpu")
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
    RayGPU()
