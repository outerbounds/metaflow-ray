from metaflow import FlowSpec, Parameter, step, pypi, kubernetes, card, current, metaflow_ray
from gpu_profile import gpu_profile


class RayGPTJFlow(FlowSpec):
    batch_size = Parameter(
        name="batch_size",
        type=int,
        default=16,
        help="batch size"
    )
    num_epochs = Parameter(
        name="num_epochs",
        type=int,
        default=2,
        help="number of epochs"
    )
    max_steps = Parameter(
        name="max_steps",
        type=int,
        default=10,
        help="number of steps at max"
    )

    def _do_ray_job(self):
        import ray
        import ray.data
        from datasets import load_dataset
        from ray.train.torch import TorchTrainer
        from ray.air import RunConfig, ScalingConfig

        from trainer import train_func
        from dataloader import split_text, tokenize

        ray.init()

        current_dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
        train_dataset = ray.data.from_huggingface(current_dataset['train']).map_batches(split_text, batch_format="pandas").map_batches(tokenize, batch_format="pandas")
        eval_dataset = ray.data.from_huggingface(current_dataset['validation']).map_batches(split_text, batch_format="pandas").map_batches(tokenize, batch_format="pandas")

        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "epochs": self.num_epochs,
            },
            scaling_config=ScalingConfig(
                num_workers=4,
                use_gpu=True,
                resources_per_worker={"CPU": 8, "GPU": 1},
            ),
            datasets={
                "train": train_dataset,
                "evaluation": eval_dataset,
            },
            run_config=RunConfig(storage_path=current.ray_storage_path)
        )

        self.result = trainer.fit()

    @step
    def start(self):
        self.next(self.train, num_parallel=4)

    @gpu_profile(interval=1)
    @kubernetes(
        image="registry.hub.docker.com/rayproject/ray:2.40.0-py311-gpu",
        cpu=8,
        gpu=1,
        memory=60000,
        shared_memory=12000,
        use_tmpfs=True
    )
    @metaflow_ray
    @pypi(
        packages={
            "torch": "2.5.1",
            "matplotlib": "3.9.3",
            "pandas": "2.2.3",
            "datasets": "3.2.0",
            "evaluate": "0.4.3",
            "transformers": "4.47.1",
            "accelerate": "1.2.1",
            "deepspeed": "0.16.2",
            "setuptools": "75.6.0",
        }
    )
    @step
    def train(self):
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
        self.metrics_df = self.result.metrics_dataframe


if __name__ == "__main__":
    RayGPTJFlow()
