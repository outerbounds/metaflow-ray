from typing import Any, Dict
from functools import partial
from gpu_profile import gpu_profile
from metaflow import FlowSpec, Parameter, step, pypi, kubernetes, card, current, metaflow_ray


def train_func(config: Dict[str, Any], batch_size: int, num_epochs: int):
    import os
    import lightning.pytorch as pl
    from model import MNISTClassifier
    from dataloader import MNISTDataModule
    from ray.train.lightning import (
        prepare_trainer,
        RayLightningEnvironment,
        RayTrainReportCallback,
        RayDDPStrategy
    )

    model = MNISTClassifier(config=config)
    dm = MNISTDataModule(batch_size=batch_size)
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(), name="tune-ptl-example", version="."
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        logger=logger,
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


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

    def _do_ray_job(self):
        import ray
        from ray import tune, air
        from ray.train.torch import TorchTrainer
        from ray.tune.schedulers import ASHAScheduler

        ray.init()

        config = {
            "layer_1_size": tune.choice([32, 64, 128]),
            "layer_2_size": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
        }

        scheduler = ASHAScheduler(
            max_t=self.num_epochs,
            grace_period=1,
            reduction_factor=2
        )

        ray_trainer = TorchTrainer(
            partial(
                train_func,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
            ),
            scaling_config=air.ScalingConfig(
                num_workers=4, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1},
            ),
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": config},
            tune_config=tune.TuneConfig(
                metric="val/accuracy",
                mode="max",
                num_samples=self.num_samples,
                scheduler=scheduler,
            ),
            run_config=air.RunConfig(
                name="tune_mnist_asha",
                storage_path=current.ray_storage_path,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute="val/accuracy",
                    checkpoint_score_order="max",
                )
            ),
        )

        results = tuner.fit()
        self.best_result = results.get_best_result(metric="val/accuracy", mode="max")

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
            "ray[air]": "2.40.0",
            "matplotlib": "3.10.0",
            "lightning": "2.4.0",
            "torchmetrics": "1.6.0",
        }
    )
    @step
    def tune(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @card
    @pypi(packages={"ray[train]": "2.40.0"})
    @step
    def end(self):
        self.metrics_df = self.best_result.metrics_dataframe


if __name__ == "__main__":
    RayLightningTuneFlow()
