from metaflow import (
    FlowSpec,
    Parameter,
    step,
    pypi,
    kubernetes,
    card,
    huggingface_hub,
    model,
    JSONType,
    current,
    metaflow_ray
)


class vLLMMistralInference(FlowSpec):
    messages = Parameter(
        name="messages",
        type=JSONType,
        required=True,
        help="messages in json format"
    )

    @step
    def start(self):
        self.next(self.pull_model_from_huggingface)

    @pypi(
        python="3.10",
        packages={
            "huggingface-hub": "0.27.0",
        }
    )
    @huggingface_hub
    @step
    def pull_model_from_huggingface(self):
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
        )
        self.next(self.run_vllm, num_parallel=2)

    @card
    @model(
        # this ensures that all workers have the model loaded to the same path
        load=[("llama_model", "./llama_model")]
    )
    @kubernetes(
        cpu=16,
        gpu=8,
        memory=60000,
        node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.9",
        shared_memory=12000
    )
    @metaflow_ray(
        # 20 mins timeout for all workers to join
        all_nodes_started_timeout=1200
    )
    @pypi(
        python="3.10",
        packages={
            "vllm": "0.6.5",
            "transformers": "4.47.1",
            "huggingface-hub": "0.27.0",
        }
    )
    @step
    def run_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(current.model.loaded["llama_model"])
        llm = LLM(
            model=current.model.loaded["llama_model"],
            tensor_parallel_size=8,
            enforce_eager=True,
        )
        sampling_params = SamplingParams(temperature=0.5)
        outputs = llm.generate(
            tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True, tokenize=False
            ),
            sampling_params,
        )
        self.text = outputs[0].outputs[0].text.strip()

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    vLLMMistralInference()
