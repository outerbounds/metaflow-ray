# Introduction

The following two files showcase how to run inference on Mistral 7B Instruct using the `@metaflow_ray` decorator with `@kubernetes`.

1. `flow.py` showcases the usage of `@huggingface_hub` to download the model along with using `@model` to ensure all workers have the model artifact loaded on a pre-defined path. The messages to be sent are a parameter to the flow and `vllm` is used for inference.

2. `main.py` runs `flow.py` on the Outerbounds platform with fast-bakery using the [Runner API](https://docs.metaflow.org/api/runner). This makes it easy to pass JSON based parameters without using the CLI.

- The flow can be run using `python main.py`
