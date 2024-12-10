from metaflow import (
    FlowSpec,
    step,
    card,
    pypi,
    kubernetes,
    parallel,
    secrets,
    huggingface_hub,
    gpu_profile,
    current,
    model,
    metaflow_ray,
)

DISK_SIZE = 100 * 1000  # 100 GB

MEMORY = 60 * 1000  # 60 GB


class RayPlacementGroupExampleFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.pull_model_from_huggingface)

    
    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.1",
            "transformers": "4.44.2",
            "huggingface-hub": "0.25.1",
        },
    )
    @huggingface_hub
    @step
    def pull_model_from_huggingface(self):
        self.model_id = "unsloth/Llama-3.2-1B-Instruct"
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
        )
        self.embedding_model_name = "intfloat/e5-mistral-7b-instruct"
        self.embedding_model = current.huggingface_hub.snapshot_download(
            repo_id=self.embedding_model_name,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
        )
        self.next(self.run_vllm, num_parallel=2)

    @gpu_profile
    @model(
        load=[("llama_model", "./llama_model"), ("embedding_model", "./embedding_model")]
    )  # this ensures that all workers have the model loaded to the same path
    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.1",
            "transformers": "4.44.2",
            "huggingface-hub": "0.25.1",
            "setuptools": "74.1.2",
        },
    )
    @kubernetes(
        cpu=16,
        gpu=2,
        memory=MEMORY,
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.9",
        shared_memory=12 * 1000,  # 12 GB shared memory as ray requires this.
        # compute_pool="obp-gpu-mega-12"
        # node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",
    )
    @metaflow_ray(
        all_nodes_started_timeout=20 * 60
    )  # 20 minute timeout so that all workers start.
    @card
    @step
    def run_vllm(self):
        
        from remote import process_prompts_with_vllm
        embeddings = process_prompts_with_vllm(
            ["What is the capital of France?", "Tell me a joke.", "Explain quantum computing."],
            "./llama_model",
            "./embedding_model",
            {"CPU": 2, "GPU": 2},
            {"CPU": 1, "GPU": 1},
        )
        print(f"Collected all embeddings of len {len(embeddings)}")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("ending the flow")


if __name__ == "__main__":
    RayPlacementGroupExampleFlow()
