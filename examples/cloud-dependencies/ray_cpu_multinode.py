from metaflow import FlowSpec, step, metaflow_ray, batch, current

N_NODES = 2
N_CPU = 8
MEMORY = 12228


class RayCPU(FlowSpec):
    def _do_ray_job(self):
        import ray

        ray.init()
        print("Ray initialized in the %s step." % current.step_name)
        print("Ray nodes: ", ray.nodes())
        print("Ray cluster resources:")
        for k, v in ray.cluster_resources().items():
            if "memory" in k.lower():
                print("%s: %sGB" % (k, round(int(v) / (1024 * 1024 * 1024), 2)))
            else:
                print("%s: %s" % (k, v))

    @step
    def start(self):
        self.next(self.big_step, num_parallel=N_NODES)

    @batch(image="rayproject/ray", cpu=N_CPU, memory=MEMORY)
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
    RayCPU()
