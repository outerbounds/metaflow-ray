from metaflow import FlowSpec, step, metaflow_ray, batch, pypi


class TestImport(FlowSpec):
    @step
    def start(self):
        self.next(self.foo, num_parallel=2)

    @pypi(packages={"ray": "2.6.3"})
    @batch
    @metaflow_ray
    @step
    def foo(self):
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TestImport()
