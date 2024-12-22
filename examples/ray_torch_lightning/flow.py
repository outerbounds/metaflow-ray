from gpu_profile import gpu_profile
from metaflow import FlowSpec, Parameter, step, pypi, kubernetes, metaflow_ray


class RayLightningTuneFlow(FlowSpec):
    num_epochs = Parameter(
        name="num_epochs",
        default=5,
        help="Number of epochs for training"
    )
    num_samples = Parameter(
        name="num_samples",
        default=10,
        help="Number of samples from parameter space"
    )
    batch_size = Parameter(
        name="batch_size",
        default=64,
        help="Batch size"
    )

    def tune_mnist_asha(self, trainer, config, scheduler, number_samples=10):
        from ray import air, tune

        tuner = tune.Tuner(
            trainer,
            param_space={"lightning_config": config},
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

    def _do_ray_job(self):
        import os
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
        from ray.train.lightning import LightningTrainer, LightningConfigBuilder

        from model import MNISTClassifier
        from dataloader import MNISTDataModule

        from pytorch_lightning.loggers import TensorBoardLogger

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
            num_workers=4,
            use_gpu=True,
            resources_per_worker={"CPU": 7, "GPU": 1},
        )

        lightning_trainer = LightningTrainer(
            lightning_config=static_lightning_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        analysis = self.tune_mnist_asha(
            trainer=lightning_trainer,
            config=searchable_lightning_config,
            scheduler=scheduler,
            number_samples=self.num_samples
        )

        print(analysis)

    @step
    def start(self):
        self.next(self.tune, num_parallel=4)

    @gpu_profile(interval=1)
    @kubernetes(memory=12228, cpu=8, gpu=1)
    @metaflow_ray
    @pypi(
        python="3.10",
        packages=
        {
            "torch": "2.5.1",
            "torchvision": "0.20.1",
            "ray[tune,train]": "2.40.0",
            "pandas": "2.2.3",
            "matplotlib": "3.10.0",
            "pytorch-lightning": "2.4.0",
            "torchmetrics": "1.6.0",
            "tensorboard": "2.18.0",
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
    RayLightningTuneFlow()
