from metaflow import (
    FlowSpec,
    step,
    trigger_on_finish,
    current,
    Parameter,
    Flow,
    Run,
    pypi,
)
from base import TabularBatchPrediction

PRODUCTION_THRESHOLD = 0.95
PARENT_FLOW_1 = "Train"
PARENT_FLOW_2 = "Tune"
TRIGGERS = [PARENT_FLOW_1, PARENT_FLOW_2]
COMMON_PKGS = {
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


@trigger_on_finish(flows=TRIGGERS)
class Score(FlowSpec, TabularBatchPrediction):
    upstream_flow = Parameter(
        "upstream", help="Upstream flow name", default=TRIGGERS[0], type=str
    )

    def _fetch_eval_set(self):
        _, valid_dataset, test_dataset = self.load_dataset()
        true_targets = valid_dataset.select_columns(cols=["target"]).to_pandas()
        return true_targets, test_dataset

    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        import pandas as pd

        self.setup()
        try:
            upstream_run = current.trigger.run
        except AttributeError:
            print("Current run was not triggered.")
            upstream_run = Flow(self.upstream_flow).latest_successful_run
        self.upstream_run_pathspec = upstream_run.pathspec

        true_targets, test_dataset = self._fetch_eval_set()
        preds = self.batch_predict(
            dataset=test_dataset, checkpoint=self.load_checkpoint(run=upstream_run)
        ).to_pandas()
        self.score_results = pd.concat([true_targets, preds], axis=1)
        for threshold in [0.25, 0.5, 0.75]:
            self.score_results[f"pred @ {threshold}"] = self.score_results[
                "predictions"
            ].apply(lambda x: 1 if x > threshold else 0)
            print(
                "Accuracy with threshold @ {threshold}: {val}%".format(
                    threshold=threshold,
                    val=round(
                        100
                        * (
                            self.score_results["target"]
                            == self.score_results[f"pred @ {threshold}"]
                        ).sum()
                        / len(self.score_results),
                        2,
                    ),
                )
            )
        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        run = Run(self.upstream_run_pathspec)
        df = self.score_results
        accuracy = (
            (df["predictions"] > 0.5).values == df["target"].values
        ).sum() / len(df)
        if accuracy > PRODUCTION_THRESHOLD:
            run = Run(self.upstream_run_pathspec)
            run.add_tag("production_ready")

        print(
            f"""
            Access result:

                from metaflow import Run
                run = Run('{current.flow_name}/{current.run_id}')
                df = run.data.score_results

            Access upstream run:

                from metaflow import Run
                run = Run('{self.upstream_run_pathspec}')
                df = run.data.result

            Fetch latest production ready runs based on tags from this step:
                
                from metaflow import Flow
                training_run = list(Flow('Train').runs('production_ready'))[0]
                tuning_run = list(Flow('Tune').runs('production_ready'))[0]
        """
        )


if __name__ == "__main__":
    Score()
