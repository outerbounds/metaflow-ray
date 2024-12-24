# Introduction

The following two files showcase how to run inference on models (default: Unsloth Llama-3.2-3B Instruct) using the `@metaflow_ray` decorator with `@kubernetes`.

1. `flow.py` showcases the usage of `@huggingface_hub` to download the model along with using `@model` to ensure all workers have the model artifact loaded on a pre-defined path. The messages to be sent are a parameter to the flow and `vllm` is used for inference.

2. `main.py` runs `flow.py` on the Outerbounds platform with fast-bakery using the [Runner API](https://docs.metaflow.org/api/runner). This makes it easy to pass JSON based parameters without using the CLI.

- The flow can be run using `python main.py` or perhaps
- `python flow.py --no-pylint --environment=fast-bakery run`

The model can be changed using the `--model_id` parameter (eg: `--model_id "mistralai/Mistral-7B-Instruct-v0.1"`).
Please make sure to have right credentials for pulling from HuggingFace.
