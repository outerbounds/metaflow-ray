from metaflow import FlowSpec, step, current


class HelloRay(FlowSpec):
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
        self._do_ray_job()
        self.next(self.end)

    @step
    def end(self):
        self._do_ray_job()


if __name__ == "__main__":
    HelloRay()
