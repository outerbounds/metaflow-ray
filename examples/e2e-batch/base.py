from typing import Callable, List, Dict, Union
import ray
from ray import tune
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors import StandardScaler
from ray.air.config import ScalingConfig
from ray.air import RunConfig
from ray.train.xgboost import XGBoostTrainer, XGBoostCheckpoint, XGBoostPredictor
from ray.train.batch_predictor import BatchPredictor
from metaflow import Run, Flow
from metaflow.metaflow_config import DATATOOLS_S3ROOT


class LearningTask:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        preprocessor: Preprocessor,
        model_name: str,
        model_hyperparameters: dict,
        session_metrics_keys: List,
        max_timeout: int,
        max_memory: int,
        n_cpu: int,
        n_gpu: int,
        n_nodes: int
        # generative or discriminative?
        # supervised or unsupervised?
        # classification or regression?
        # offline or online?
        # batch, streaming, realtime?
    ):

        # Idea:
        # 1 dataset per learning task,
        # 1 model per learning task,
        # many sessions per learning task.

        # ray.data.Dataset
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        # ray.data.preprocessor.Preprocessor
        self.preprocessor = preprocessor

        # ray.train.trainer.BaseTrainer
        self.model_name = model_name
        self.model_hyperparameters = model_hyperparameters

        # ray.air.session metrics
        self.session_metrics_keys: List = session_metrics_keys

        # task limits
        self.max_timeout = max_timeout
        self.max_memory = max_memory
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.n_nodes = n_nodes

    def load_dataset(self):
        raise NotImplementedError(
            "load_dataset not implemented. LearningTask is an abstract class."
        )

    def load_preprocesser(self):
        raise NotImplementedError(
            "process_features not implemented. LearningTask is an abstract class."
        )

    def load_checkpoint(self):
        raise NotImplementedError(
            "load_checkpoint not implemented. LearningTask is an abstract class."
        )

    def load_trainer(self):
        raise NotImplementedError(
            "load_trainer not implemented. LearningTask is an abstract class."
        )

    def load_tuner(self):
        raise NotImplementedError(
            "train_model not implemented. LearningTask is an abstract class."
        )

    def batch_predict(self):
        raise NotImplementedError(
            "predict_batch not implemented. LearningTask is an abstract class."
        )


class TabularBatchPrediction(LearningTask):
    def __init__(self, kwargs: dict = {}):
        self.setup(**kwargs)

    def setup(
        self,
        dataset_name: str = None,
        dataset_path: str = None,
        preprocessor: Preprocessor = None,
        model_name: str = "XGBoost Classifier",
        model_hyperparameters: dict = {},
        session_metrics_keys: List = ["accuracy"],
        max_timeout: int = 3600,
        max_memory: int = 8000,
        n_cpu: int = 4,
        n_gpu: int = 0,
        n_nodes: int = 1,
    ):

        if dataset_name is None:
            print(
                "No dataset name or loader provided. Using default breast_cancer.csv dataset."
            )
            dataset_name = "breast_cancer"
            dataset_path = "s3://anonymous@air-example-data/breast_cancer.csv"

        if preprocessor is None and dataset_name == "breast_cancer":
            columns_to_scale = ["mean radius", "mean texture"]
            preprocessor = StandardScaler(columns=columns_to_scale)

        super().__init__(
            dataset_name,
            dataset_path,
            preprocessor,
            model_name,
            model_hyperparameters,
            session_metrics_keys,
            max_timeout,
            max_memory,
            n_cpu,
            n_gpu,
            n_nodes,
        )

        self.hpo_num_samples = 5

    def load_dataset(
        self, path: str = None, test_size: float = 0.3
    ) -> List[ray.data.Dataset]:
        if path is None:
            assert self.dataset_path is not None, "No dataset path provided."
            path = self.dataset_path
        dataset = ray.data.read_csv(path)
        train_dataset, valid_dataset = dataset.train_test_split(test_size=test_size)
        test_dataset = valid_dataset.drop_columns(["target"])
        return train_dataset, valid_dataset, test_dataset

    def load_preprocesser(self):
        columns_to_scale = ["mean radius", "mean texture"]
        return StandardScaler(columns=columns_to_scale)

    def load_checkpoint(
        self, run: Run = None, flow_name: str = "Train"
    ) -> ray.train.xgboost.XGBoostCheckpoint:
        if run is None:
            run = Flow(flow_name).latest_successful_run
        print("Loading Ray checkpoint from {}/{}".format(flow_name, run.id))
        return run.data.result.checkpoint

    def load_trainer(self, trainer_args: dict = {}) -> ray.train.xgboost.XGBoostTrainer:
        train_dataset, valid_dataset, _ = self.load_dataset()
        preprocessor = self.load_preprocesser()
        _trainer_args = dict(
            run_config=RunConfig(),
            scaling_config=ScalingConfig(
                num_workers=self.n_nodes,
                use_gpu=self.n_gpu > 0,
                _max_cpu_fraction_per_node=0.9,
            ),
            label_column="target",
            params=dict(
                tree_method="approx",
                objective="binary:logistic",
                eval_metric=["logloss", "error"],
                max_depth=2,
            ),
            datasets={"train": train_dataset, "valid": valid_dataset},
            preprocessor=preprocessor,
            num_boost_round=5,
        )
        _trainer_args.update(trainer_args)
        return XGBoostTrainer(**_trainer_args)

    def load_tuner(self, tuner_args: dict = {}):

        trainer = self.load_trainer()

        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html#ray.tune.Tuner
        param_space = {
            "scaling_config": ScalingConfig(
                num_workers=1,
                # resources_per_worker={"CPU": 1},
                _max_cpu_fraction_per_node=0.8,
            ),
            "params": {
                "objective": "binary:logistic",
                "tree_method": "approx",
                "eval_metric": ["logloss", "error"],
                "eta": tune.loguniform(1e-4, 1e-1),
                "subsample": tune.uniform(0.5, 1.0),
                "max_depth": tune.randint(1, 9),
            },
        }

        run_config = RunConfig(verbose=0)

        _num_samples = self.num_samples if hasattr(self, 'num_samples') else self.hpo_num_samples
        print(f"Using {_num_samples} samples for HPO.")
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
        tune_config = tune.TuneConfig(
            metric="valid-logloss", 
            mode="min",
            search_alg=tune.search.basic_variant.BasicVariantGenerator(),
            scheduler=tune.schedulers.ASHAScheduler(),
            num_samples = _num_samples,
            time_budget_s = self.max_timeout,
            max_concurrent_trials = self.n_nodes,
        )

        _tuner_args = dict(
            trainable=trainer,
            param_space=param_space,
            run_config=run_config,
            tune_config=tune_config,
        )

        _tuner_args.update(tuner_args)
        return ray.tune.Tuner(**_tuner_args)

    def batch_predict(
        self,
        dataset: ray.data.dataset.Dataset,
        checkpoint: XGBoostCheckpoint = None,
        load_checkpoint_args: dict = {},
    ) -> ray.data.dataset.Dataset:
        if checkpoint is None:
            checkpoint = self.load_checkpoint(**load_checkpoint_args)
        batch_predictor = BatchPredictor.from_checkpoint(checkpoint, XGBoostPredictor)
        return batch_predictor.predict(dataset)

    def choose_best_checkpoint(
        self, run: Run = None, flow_name: str = "Train"
    ) -> XGBoostCheckpoint:
        def _search_for_checkpt(run):
            if run.data.result is None:
                return None
            if run.data.result.checkpoint is None:
                return None
            elif isinstance(run.data.result.checkpoint, XGBoostCheckpoint):
                return run.data.result.checkpoint

        if run is None:
            for run in Flow(flow_name):
                chkpt = _search_for_checkpt(run)
                if chkpt:
                    return chkpt
        else:
            chkpt = _search_for_checkpt(run)
            if chkpt:
                return chkpt

        raise ValueError("No checkpoint found in {}/{}".format(flow_name, run_id))
