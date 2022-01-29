from typing import Any
import gokart
from .datamanager.task import BuildTrainDataset
from .features.task import TrainFeatureModel, ApplyFeatureModel


class TrainWorkflow(gokart.TaskOnKart):
    """
    Workflow
    """

    done = False

    def requires(self) -> Any:
        dataset: gokart.TaskOnKart = BuildTrainDataset()
        feature_model: gokart.TaskOnKart = TrainFeatureModel(task_dataset=dataset)
        feature: gokart.TaskOnKart = ApplyFeatureModel(
            task_dataset=dataset, task_feature_model=feature_model
        )

        return feature

    def run(self) -> None:
        print("INFO: Running wrokflow...")
        self.done = True

    def complete(self) -> bool:
        return self.done
