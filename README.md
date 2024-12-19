# Metaflow-Ray

### Introduction
`metaflow-ray` is an extension for Metaflow that enables seamless integration with Ray, allowing users to easily leverage 
Ray's powerful distributed computing capabilities within their Metaflow flows. With `metaflow-ray`, you can spin up ephemeral Ray clusters on AWS Batch or Kubernetes directly from your Metaflow steps using the `@metaflow_ray` decorator. This enables you to run your Ray applications that leverage Ray Core, Ray Train, Ray Tune, and Ray Data effortlessly within your Metaflow flow.

### Features
- <b>Effortless Ray Integration:</b> This extension provides a simple and intuitive way to incorporate Ray 
into your Metaflow workflows using the `@metaflow_ray` decorator.
- <b>Elastic Ephemeral Ray Clusters:</b> Let Metaflow orchestrate the creation of ephemeral Ray clusters on top of either:
    - AWS Batch multi-node parallel jobs
    - Kubernetes JobSets
- <b>Seamless Ray Initialization:</b> The `@metaflow_ray` decorator handles the initialization of the Ray cluster for you, so you can focus on writing your Ray code without worrying about cluster setup
- <b>Wide Range of Applications:</b> Run a wide variety of Ray applications, including hyperparameter tuning, distributed data processing, and distributed training, etc.

### Installation
You can install `metaflow-ray` via `pip` alongside your existing Metaflow installation:
```
pip install metaflow-ray
```

### Getting Started
1. Import the `@metaflow_ray` decorator to enable integration:

```python
from metaflow import metaflow_ray
```

2. Decorate your step with `@metaflow_ray` and Initialize Ray within Your Step:

```python
@step
def start(self):
    self.next(self.train, num_parallel=NUM_NODES)

@metaflow_ray
@pypi(packages={"ray": "2.39.0"})
@batch(**RESOURCES) # You can even use @kubernetes 
@step
def train(self):
    import ray
    ray.init()
    # Your step's training code here

    self.next(self.join)

@step
def join(self, inputs):
    self.next(self.end)

@step
def end(self):
    pass
```

### Some things to consider:

1. The `num_parallel` argument must always be specified in the step preceding the transition to a step decorated with `@metaflow_ray`. In the example above, the `start` step transitions to the `train` step, and it includes the `num_parallel` argument because the `train` step is decorated with `@metaflow_ray`. This ensures the `train` step can execute in parallel as intended.
- As a consequence, there must always exist a corresponding `join` step as highlighted in the snippet above.

2. For remote execution environments (i.e. `@metaflow_ray` is used in conjunction with `@batch` or `@kubernetes`), the value of `num_parallel` should greater than 1 i.e. at least 2. However, when using the `@metaflow_ray` decorator in a standalone manner, the value of `num_parallel` cannot be greater than 1 (on Windows and macOS) because locally spun up ray clusters do not support multiple nodes unless the underlying OS is linux based.
- Ideally, `ray` should be available in the remote execution environments. If not, one can always use the `@pypi` decorator to introduce `ray` as a dependency.

4. If the `@metaflow_ray` decorator is used in a local context i.e. without `@batch` or `@kubernetes`, a local ray cluster is spinned up, provided that the `ray` library (installable via `pip install ray`) is available in the underlying python environment. Running the flow again (locally) could result in the issue of:
```
ConnectionError: Ray is trying to start at 127.0.0.1:6379, but is already running at 127.0.0.1:6379.
Please specify a different port using the `--port` flag of `ray start` command.
```
One can simply run `ray stop` in another terminal to terminate the ray cluster that was spun up locally.

### Examples
Check out the [examples](/examples) directory for sample Metaflow flows that demonstrate how to use the `metaflow-ray` extension 
with various Ray applications.

| Directory | Description |
| :--- | ---: |
| [Counter!](examples/basic_counter/README.md) | Run a basic Counter with Ray that increments in Python, then do it inside a Metaflow task! |
| [Process Dataframe!](examples/dataframe_process/README.md) | Process a large dataframe in chunks with Ray and Python, then do it inside a Metaflow task! |
| [Custom Docker Images!](examples/custom_docker_images/README.md) | Specify custom docker images on kubernetes / batch with Ray on Metaflow |
| [Train XGBoost](examples/train_xgboost/README.md) | Use [Ray Train](https://docs.ray.io/en/latest/train/train.html) to build XGBoost models on multiple nodes, including CPU and GPU examples. |
| [Tune PyTorch](examples/tune/README.md) | Use [Ray Tune](https://docs.ray.io/en/latest/tune/tune.html) to build PyTorch models on one or more nodes, including CPU and GPU examples. |  
| [End-to-end Batch Workflow](examples/e2e-batch/README.md) | Train models, evaluate them, and serve them. See how to use Metaflow workflows and various Ray abstractions together in a complete workflow. |
| [PyTorch Lightning](examples/ray-lightning-tune/README.md) | Get started run a PyTorch Lightning job on the Ray cluster formed in a `@metaflow_ray` step. |
| [GPT-J Fine Tuning](examples/ray-fine-tuning-gpt-j/README.md) | Fine tune the 6B parameter GPT-J model on a Ray cluster. |

### License
`metaflow-ray` is distributed under the <u>Apache License</u>.