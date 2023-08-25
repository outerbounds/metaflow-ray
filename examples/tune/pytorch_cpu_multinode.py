from metaflow import FlowSpec, step, batch, current, card, ray_parallel, conda
from metaflow.cards import Image
from metaflow.metaflow_config import DATATOOLS_S3ROOT

NUM_NODES = 2
RESOURCES = dict(memory=12228, cpu=4)
CONDA_DEP = dict(
    libraries={"pytorch::pytorch": "2.0.1", "pytorch::torchvision": "0.15.2"},
    pip_packages={"ray[air]": "", "pandas": "2.0.3", "matplotlib": "3.7.2"},
)


class RayTorchMultinodeCPU(FlowSpec):

    epoch_size = 1024
    test_size = 256
    num_samples = 20
    num_workers = NUM_NODES
    n_cpu = RESOURCES["cpu"]

    @step
    def start(self):
        self.next(self.tune, num_parallel=NUM_NODES)

    @conda(**CONDA_DEP)
    @ray_parallel
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
        from ray import tune
        from ray.air.config import ScalingConfig

        ray.init()
        search_space = {
            "lr": ray.tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": ray.tune.uniform(0.1, 0.9),
            "scaling_config": ScalingConfig(
                num_workers=self.num_workers,
                resources_per_worker={"CPU": self.n_cpu},
                _max_cpu_fraction_per_node=0.8,
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

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayTorchMultinodeCPU()
