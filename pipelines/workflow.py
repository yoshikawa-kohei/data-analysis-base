from typing import Any, List, Type
import gokart
import luigi
from .datamanager.task import BuildTrainDataset, BuildTestDataset
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
        self.dump(self.load())

    def complete(self) -> bool:
        return self.done


class TestWorkflow(gokart.TaskOnKart):
    """
    Workflow
    """

    done = False

    def requires(self) -> Any:
        train_dataset: gokart.TaskOnKart = BuildTrainDataset()
        test_dataset: gokart.TaskOnKart = BuildTestDataset()
        feature_model: gokart.TaskOnKart = TrainFeatureModel(task_dataset=train_dataset)
        feature: gokart.TaskOnKart = ApplyFeatureModel(
            task_dataset=test_dataset, task_feature_model=feature_model
        )

        return feature

    def run(self) -> None:
        print("INFO: Running wrokflow...")
        self.done = True
        self.dump(self.load())

    def complete(self) -> bool:
        return self.done

