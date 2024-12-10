from metaflow import FlowSpec, step, kubernetes, metaflow_ray, pypi
from gpu_profile import gpu_profile


class RayGPUFlow(FlowSpec):

    def _do_ray_job(self):
        import ray

        ray.init()

        print("number of ray nodes: ", ray.nodes())
        memory = ray.cluster_resources().get("memory")
        print("memory: %sGB" % (round(int(memory) / (1024 * 1024 * 1024), 2)))
    
    @step
    def start(self):
        self.next(self.execute, num_parallel=2)

    @gpu_profile(interval=1)
    @pypi(packages={"matplotlib": "3.9.3"})
    @kubernetes(image="registry.hub.docker.com/rayproject/ray:latest-gpu", gpu=1, memory=12228)
    @metaflow_ray
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
    RayGPUFlow()
