from fastapi import FastAPI
import ray
from ray import serve
from metaflow import Flow
from ray.train.xgboost import XGBoostPredictor
from ray.train.batch_predictor import BatchPredictor
from typing import List, Dict

app = FastAPI()


def select_from_checkpoint_registry(flow_name="Train"):
    from metaflow import Flow

    run = Flow(flow_name).latest_successful_run
    print("Using checkpoint from Run('{}')".format(run.pathspec))
    result = run.data.result
    return result.checkpoint


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class BatchPredictionService:
    def __init__(self):
        checkpoint = select_from_checkpoint_registry()
        self.predictor = BatchPredictor.from_checkpoint(checkpoint, XGBoostPredictor)

    @app.post("/")
    def hello(self) -> Dict[str, str]:
        print("Hello Server Logs!")
        return {"response": "Hello World!"}

    @app.post("/predict/")
    def predict(
        self, id_to_batch_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        features = ray.data.from_items(list(id_to_batch_features.values()))

        # CHANGE THIS TO USE NEW PREDICTOR
        preds = self.predictor.predict(features).to_pandas()["predictions"].values

        id_to_preds_payload = dict(zip(id_to_batch_features.keys(), preds))
        return id_to_preds_payload

    @app.post("/swap-model/")
    def swap_model(self) -> Dict[str, str]:
        raise NotImplementedError
        return {"response": "TODO"}


batch_preds = BatchPredictionService.bind()
