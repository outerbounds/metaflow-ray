from metaflow import FlowSpec, Parameter, step, pypi, kubernetes, current, metaflow_ray
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
        default=1,
        help="number of epochs"
    )

    def _do_ray_job(self):
        import ray
        import ray.data
        from datasets import load_dataset
        from ray.data.preprocessors import Chain
        from ray.air import RunConfig, ScalingConfig
        from ray.data.preprocessors import BatchMapper
        from ray.train.huggingface import TransformersTrainer

        from dataloader import split_text, tokenize
        from trainer import trainer_init_per_worker

        ray.init()

        current_dataset = load_dataset("tiny_shakespeare")
        ray_datasets = ray.data.from_huggingface(current_dataset)
        splitter = BatchMapper(split_text, batch_format="pandas")
        tokenizer = BatchMapper(tokenize, batch_format="pandas")

        trainer = TransformersTrainer(
            trainer_init_per_worker=trainer_init_per_worker,
            trainer_init_config={
                "batch_size": self.batch_size,
                "epochs": self.num_epochs,
            },
            scaling_config=ScalingConfig(
                num_workers=4,
                use_gpu=True
            ),
            datasets={
                "train": ray_datasets["train"],
                "evaluation": ray_datasets["validation"],
            },
            preprocessor=Chain(splitter, tokenizer),
            run_config=RunConfig(storage_path=current.ray_storage_path),
        )

        results = trainer.fit()
        print(results)
        checkpoint = results.checkpoint
        print(checkpoint)

    @step
    def start(self):
        self.next(self.train, num_parallel=4)

    @gpu_profile(interval=1)
    @kubernetes(cpu=8, gpu=1, memory=60000, shared_memory=12000, use_tmpfs=True)
    @metaflow_ray
    @pypi(
        python="3.10",
        packages={
            "matplotlib": "3.9.3",
            "pandas": "2.2.3",
            "datasets": "3.2.0",
            "evaluate": "0.4.3",
            "transformers": "4.47.1",
            "ray[air]": "2.40.0"
        }
    )
    @step
    def train(self):
        self._do_ray_job()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayGPTJFlow()
