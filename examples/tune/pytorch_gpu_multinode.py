from metaflow import (
    FlowSpec,
    step,
    batch,
    S3,
    pypi,
    current,
    card,
    metaflow_ray,
    conda,
    kubernetes,
)
from metaflow.cards import Image
from metaflow.metaflow_config import DATATOOLS_S3ROOT
from decorators import gpu_profile

NUM_NODES = 4
RESOURCES = dict(memory=12228, cpu=8, gpu=1)
COMMON_PKGS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "ray[tune]": "2.7.1",
    "pandas": "2.1.0",
    "matplotlib": "3.7.2",
    "pyarrow": "13.0.0",
}


class RayTorchMultinodeGPU(FlowSpec):
    epoch_size = 1024
    test_size = 256
    num_samples = 20
    num_workers = NUM_NODES
    n_cpu = 1  # RESOURCES['cpu']
    n_gpu = 0.25  # RESOURCES['gpu']

    @pypi(packages=COMMON_PKGS, python="3.9.10")
    @step
    def start(self):
        self.next(self.tune, num_parallel=NUM_NODES)

    @pypi(packages=COMMON_PKGS, python="3.9.10")
    @gpu_profile(interval=1)
    @batch(**RESOURCES)
    @metaflow_ray
    @card
    @step
    def tune(self):
        from pytorch_example import train_mnist, run, plot
        from matplotlib import pyplot as plt
        from functools import partial
        import ray
        import pandas as pd
        import numpy as np
        import os
        from ray.air.config import ScalingConfig

        ray.init()
        search_space = {
            "lr": ray.tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": ray.tune.uniform(0.1, 0.9),
            "scaling_config": ScalingConfig(
                use_gpu=self.n_gpu > 0,
                num_workers=self.num_workers,
                resources_per_worker={"CPU": self.n_cpu, "GPU": self.n_gpu},
            ),
        }

        results_list = run(
            search_space=search_space,
            smoke_test=True,
            run_config_storage_path=self.checkpoint_path,
        )

        fig, ax = plt.subplots(1, 1)
        result_dfs = plot(results_list, ax=ax)
        self.result = pd.concat(list(result_dfs.values()))
        current.card.append(Image.from_matplotlib(fig))

        self.next(self.join)

    @pypi(packages=COMMON_PKGS, python="3.9.10")
    @step
    def join(self, inputs):
        self.next(self.end)

    @pypi(packages=COMMON_PKGS, python="3.9.10")
    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayTorchMultinodeGPU()
