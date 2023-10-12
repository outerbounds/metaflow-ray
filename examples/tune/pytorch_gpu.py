from metaflow import FlowSpec, step, batch, Parameter, S3, current, pypi, card
from metaflow.cards import Image
from metaflow.metaflow_config import DATATOOLS_S3ROOT
from decorators import gpu_profile

RESOURCES = dict(memory=12228, cpu=2, gpu=1)
COMMON_PKGS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "matplotlib": "3.7.2",
    "pyarrow": "13.0.0",
}


class RayTorchGPU(FlowSpec):
    epoch_size = 1024
    test_size = 256
    num_samples = 20
    n_cpu = RESOURCES["cpu"]
    n_gpu = RESOURCES["gpu"]

    @step
    def start(self):
        self.next(self.tune)

    @pypi(packages=COMMON_PKGS)
    @gpu_profile(interval=1)
    @batch(**RESOURCES)
    @card
    @step
    def tune(self):
        from functools import partial
        from pytorch_example import train_mnist, run, plot
        from matplotlib import pyplot as plt
        import pandas as pd
        import numpy as np
        import ray
        import os
        from ray.air.config import ScalingConfig

        ray.init()
        search_space = {
            "lr": ray.tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": ray.tune.uniform(0.1, 0.9),
            "scaling_config": ScalingConfig(
                use_gpu=True,
                resources_per_worker={"CPU": self.n_cpu, "GPU": self.n_gpu},
                _max_cpu_fraction_per_node=0.8,
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
    RayTorchGPU()
