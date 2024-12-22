from metaflow.cards import Image
from metaflow import FlowSpec, Parameter, step, card, kubernetes, pypi, metaflow_ray, current


class RayTuneTorchCPU(FlowSpec):
    num_samples = 20
    batch_size = Parameter(
        name="batch_size",
        default=1024,
        type=int,
        help="batch size for training"
    )
    test_batch_size = Parameter(
        name="test_batch_size",
        default=256,
        type=int,
        help="batch size for testing"
    )
    smoke_test = Parameter(
        name="smoke_test",
        default=True,
        type=bool,
        help="exit early on a small subset of data"
    )

    def _do_ray_job(self):
        import ray
        import numpy as np
        import pandas as pd
        from utils import run, plot
        from matplotlib import pyplot as plt
        from ray.air.config import ScalingConfig

        ray.init()

        search_space = {
            "lr": ray.tune.sample_from(lambda _: 10 ** (-10 * np.random.rand())),
            "momentum": ray.tune.uniform(0.1, 0.9),
            "scaling_config": ScalingConfig(
                num_workers=2, resources_per_worker={"CPU": 4}
            ),
        }

        results_list = run(
            search_space=search_space,
            batch_size=self.batch_size,
            test_batch_size=self.test_batch_size,
            num_samples=self.num_samples,
            smoke_test=self.smoke_test,
            run_config_storage_path=current.ray_storage_path,
        )

        fig, ax = plt.subplots(1, 1)
        result_dfs = plot(results_list, ax=ax)
        self.result = pd.concat(list(result_dfs.values()))
        current.card.append(Image.from_matplotlib(fig))

    @step
    def start(self):
        self.next(self.tune, num_parallel=2)

    @card
    @kubernetes(memory=12228, cpu=4)
    @metaflow_ray
    @pypi(
        python="3.10",
        packages=
        {
            "torch": "2.5.1",
            "torchvision": "0.20.1",
            "ray[tune]": "2.40.0",
            "pandas": "2.2.3",
            "matplotlib": "3.10.0",
        }
    )
    @step
    def tune(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayTuneTorchCPU()
