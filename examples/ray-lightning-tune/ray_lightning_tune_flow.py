from metaflow import (
    FlowSpec,
    step,
    batch,
    Parameter,
    S3,
    current,
    card,
    pypi,
    metaflow_ray,
    Parameter,
    IncludeFile,
)
from decorators import gpu_profile

NUM_NODES = 4
RESOURCES = dict(memory=12228, cpu=8, gpu=1)
COMMON_PKGS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "matplotlib": "3.7.2",
    "optuna": "3.3.0",
    "pytorch-lightning": "2.0.9",
    "torchmetrics": "1.1.2",
    "pyarrow": "13.0.0",
    "tensorboard": "2.14.0",
}
# TODO [NEEDS UPGRADE] : Change this code to support the lastest version of the @metaflow_ray decorator

class RayLightningTuneFlow(FlowSpec):
    # default_config = IncludeFile(name="config", default="config.json", help="config file")

    num_epochs = Parameter(
        "num_epochs", default=5, help="Number of epochs for training"
    )
    num_samples = Parameter(
        "num_samples", default=10, help="Number of samples from parameter space"
    )
    num_worker_nodes = NUM_NODES
    n_cpu = RESOURCES["cpu"] - 1
    n_gpu = RESOURCES["gpu"]

    batch_size = Parameter("batch_size", default=64, help="Batch size")

    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.next(self.tune, num_parallel=NUM_NODES)

    @pypi(packages=COMMON_PKGS)
    @gpu_profile(interval=1)
    @metaflow_ray
    @batch(**RESOURCES)
    @card
    @step
    def tune(self):
        from pytorch_lightning.loggers import TensorBoardLogger
        import ray
        from ray import air, tune
        from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
        from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
        from ray.train.lightning import LightningTrainer, LightningConfigBuilder
        from model import MNISTClassifier
        from dataloader import MNISTDataModule
        import os

        ray.init()

        dm = MNISTDataModule(batch_size=self.batch_size)
        logger = TensorBoardLogger(
            save_dir=os.getcwd(), name="tune-ptl-example", version="."
        )

        # Static configs that does not change across trials
        static_lightning_config = (
            LightningConfigBuilder()
            .module(cls=MNISTClassifier)
            .trainer(max_epochs=self.num_epochs, accelerator="gpu", logger=logger)
            .fit_params(datamodule=dm)
            .checkpointing(monitor="ptl/val_accuracy", save_top_k=2, mode="max")
            .build()
        )

        # Searchable configs across different trials
        searchable_lightning_config = (
            LightningConfigBuilder()
            .module(
                config={
                    "layer_1_size": tune.choice([32, 64, 128]),
                    "layer_2_size": tune.choice([64, 128, 256]),
                    "lr": tune.loguniform(1e-4, 1e-1),
                }
            )
            .build()
        )

        # Make sure to also define an AIR CheckpointConfig here
        # to properly save checkpoints in AIR format.
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )

        scheduler = ASHAScheduler(
            max_t=self.num_epochs, grace_period=1, reduction_factor=2
        )
        scaling_config = ScalingConfig(
            num_workers=self.num_worker_nodes,
            use_gpu=True,
            resources_per_worker={"CPU": self.n_cpu, "GPU": self.n_gpu},
        )

        lightning_trainer = LightningTrainer(
            lightning_config=static_lightning_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        def tune_mnist_asha(number_samples=10):
            scheduler = ASHAScheduler(
                max_t=self.num_epochs, grace_period=1, reduction_factor=2
            )

            tuner = tune.Tuner(
                lightning_trainer,
                param_space={"lightning_config": searchable_lightning_config},
                tune_config=tune.TuneConfig(
                    metric="ptl/val_accuracy",
                    mode="max",
                    num_samples=number_samples,
                    scheduler=scheduler,
                ),
                run_config=air.RunConfig(
                    name="tune_mnist_asha",
                ),
            )
            results = tuner.fit()
            best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")

            return best_result

        analysis = tune_mnist_asha(number_samples=self.num_samples)
        print(analysis)

        self.next(self.join)

    @pypi(packages=COMMON_PKGS)
    @step
    def join(self, inputs):
        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayLightningTuneFlow()
