from metaflow import Run, Flow
from typing import List, Dict, Optional


class LearningTask:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        preprocessor = None,
        model_name: Optional[str] = None,
        model_hyperparameters: Optional[Dict] = None,
        session_metrics_keys: Optional[List] = None,
        max_timeout: int = 3600,
        max_memory: int = 8000,
        n_cpu: int = 4,
        n_gpu: int = 0,
        n_nodes: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.model_hyperparameters = model_hyperparameters
        self.session_metrics_keys = session_metrics_keys
        self.max_timeout = max_timeout
        self.max_memory = max_memory
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.n_nodes = n_nodes


class TabularBatchPrediction(LearningTask):
    def __init__(self, kwargs: Optional[Dict] = None):
        if kwargs is None:
            kwargs = {}
        self.setup(**kwargs)

    def setup(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        preprocessor = None,
        model_name: Optional[str] = None,
        model_hyperparameters: Optional[Dict] = None,
        session_metrics_keys: Optional[List] = None,
        max_timeout: int = 3600,
        max_memory: int = 8000,
        n_cpu: int = 4,
        n_gpu: int = 0,
        n_nodes: int = 1
    ):
        from ray.data.preprocessors import StandardScaler


        if dataset_name is None:
            dataset_name = "breast_cancer"
            dataset_path = "s3://anonymous@air-example-data/breast_cancer.csv"

        if preprocessor is None and dataset_name == "breast_cancer":
            columns_to_scale = ["mean radius", "mean texture"]
            preprocessor = StandardScaler(columns=columns_to_scale)

        if model_name is None:
            model_name = "XGBoost Classifier"
        
        if model_hyperparameters is None:
            model_hyperparameters = {}

        if session_metrics_keys is None:
            session_metrics_keys = ["accuracy"]

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

        self.hpo_num_samples = 10

    def load_dataset(self, test_size: float = 0.3):
        from ray.data import read_csv


        assert self.dataset_path is not None, "No dataset path provided."
        dataset = read_csv(self.dataset_path)
        train_dataset, valid_dataset = dataset.train_test_split(test_size=test_size)
        test_dataset = valid_dataset.drop_columns(["target"])
        return train_dataset, valid_dataset, test_dataset

    def load_preprocesser(self):
        return self.preprocessor

    def load_checkpoint(self, run: Optional[Run] = None, flow_name: str = "Train"):
        if run is None:
            run = Flow(flow_name).latest_successful_run
        print("Loading Ray checkpoint from {}/{}".format(flow_name, run.id))
        return run.data.result.checkpoint

    def load_trainer(self, trainer_args: Optional[Dict] = None):
        from ray.air import ScalingConfig
        from ray.train.xgboost import XGBoostTrainer


        train_dataset, valid_dataset, _ = self.load_dataset()
        preprocessor = self.load_preprocesser()

        train_dataset = preprocessor.fit_transform(train_dataset)
        valid_dataset = preprocessor.transform(valid_dataset)

        default_trainer_args = dict(
            scaling_config=ScalingConfig(
                num_workers=self.n_nodes,
                use_gpu=self.n_gpu > 0,
                resources_per_worker={"CPU": self.n_cpu, "GPU": self.n_gpu},
            ),
            label_column="target",
            params=dict(
                objective="binary:logistic",
                tree_method="approx",
                eval_metric=["logloss", "error"],
                max_depth=2,
                device="cuda" if self.n_gpu > 0 else "cpu"
            ),
            datasets={"train": train_dataset, "valid": valid_dataset},
            num_boost_round=5,
        )

        if trainer_args is not None:
            default_trainer_args.update(trainer_args)

        return XGBoostTrainer(**default_trainer_args)

    def load_tuner(self,
        trainer_args: Optional[Dict] = None,
        tuner_args: Optional[Dict] = None
    ):
        from ray import tune


        trainer = self.load_trainer(trainer_args)

        search_space = {
            "params": {
                "objective": "binary:logistic",
                "tree_method": "approx",
                "eval_metric": ["logloss", "error"],
                "eta": tune.loguniform(1e-4, 1e-1),
                "subsample": tune.uniform(0.5, 1.0),
                "max_depth": tune.randint(1, 9),
            }
        }

        tune_config = tune.TuneConfig(
            metric="valid-logloss",
            mode="min",
            search_alg=tune.search.basic_variant.BasicVariantGenerator(),
            scheduler=tune.schedulers.ASHAScheduler(),
            num_samples=self.hpo_num_samples,
            time_budget_s=self.max_timeout,
            max_concurrent_trials=self.n_nodes,
        )

        default_tuner_args = dict(
            trainable=trainer,
            param_space=search_space,
            tune_config=tune_config,
        )

        if tuner_args is not None:
            default_tuner_args.update(tuner_args)

        return tune.Tuner(**default_tuner_args)

    def batch_predict(self,
        dataset,
        checkpoint,
        load_checkpoint_args: Optional[Dict] = None
    ):
        from ray.train.xgboost import XGBoostPredictor


        if load_checkpoint_args is None:
            load_checkpoint_args = {}

        if checkpoint is None:
            checkpoint = self.load_checkpoint(**load_checkpoint_args)

        predictor = XGBoostPredictor.from_checkpoint(checkpoint)
        return dataset.map_batches(predictor.predict)
