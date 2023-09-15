from metaflow import FlowSpec, step, batch, Parameter, S3, current, card, pypi
from metaflow.cards import Image

RESOURCES = dict(memory=12228, cpu=4)
CONDA_DEP = dict(
    libraries={"pytorch::pytorch": "2.0.1", "pytorch::torchvision": "0.15.2"},
    pip_packages={"ray[air]": "", "pandas": "2.0.3", "matplotlib": "3.7.2"},
)
COMMON_PKGS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "matplotlib": "3.7.2",
    "pyarrow": "13.0.0",
}


class RayTorchCPU(FlowSpec):
    epoch_size = 1024
    test_size = 256
    num_samples = 20
    n_cpu = RESOURCES["cpu"]

    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.tune)

    @pypi(packages=COMMON_PKGS)
    @batch(**RESOURCES)
    @card
    @step
    def tune(self):
        from pytorch_example import train_mnist, run, plot
        from matplotlib import pyplot as plt
        from functools import partial
        import ray
        import os
        from ray.air.config import ScalingConfig
        import pandas as pd
        import numpy as np

        ray.init()
        search_space = {
            "lr": ray.tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": ray.tune.uniform(0.1, 0.9),
            "scaling_config": ScalingConfig(
                resources_per_worker={"CPU": self.n_cpu}, _max_cpu_fraction_per_node=0.8
            ),
        }
        results_list = run(search_space=search_space, smoke_test=True)

        fig, ax = plt.subplots(1, 1)
        result_dfs = plot(results_list, ax=ax)
        self.result = pd.concat(list(result_dfs.values()))
        current.card.append(Image.from_matplotlib(fig))

        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayTorchCPU()
