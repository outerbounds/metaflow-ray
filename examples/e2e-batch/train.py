from metaflow import FlowSpec, step, current
from base import TabularBatchPrediction


class Train(FlowSpec, TabularBatchPrediction):
    @step
    def start(self):
        self.setup()
        self.result = self.load_trainer().fit()
        self.next(self.end)

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