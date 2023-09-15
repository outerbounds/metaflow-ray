from metaflow import FlowSpec, step, current, pypi, IncludeFile
from base import TabularBatchPrediction

COMMON_PKGS = {
    "ray": "2.6.3",
    "metaflow-ray": "0.0.1",
    "pandas": "2.1.0",
    "xgboost": "2.0.0",
    "xgboost-ray": "0.1.18",
    "pyarrow": "13.0.0",
    "matplotlib": "3.7.3",
}


class Train(FlowSpec, TabularBatchPrediction):
    @pypi(packages=COMMON_PKGS)
    @step
    def start(self):
        self.setup()
        self.result = self.load_trainer().fit()
        self.next(self.end)

    @pypi(packages=COMMON_PKGS)
    @step
    def end(self):
        print(
            f"""

            Access result:

            from metaflow import Run
            run = Run('{current.flow_name}/{current.run_id}')
            result = run.data.result
        """
        )


if __name__ == "__main__":
    Train()
