from typing import Any

import gokart
import luigi

from .dataset.task import BuildDataset, TestDataset, TrainDataset
from .features.task import ApplyFeatureModel, FitFeatureModel


class TrainWorkflow(gokart.TaskOnKart):
    """
    Workflow
    """

    done = False

    def requires(self) -> Any:
        org_dataset = BuildDataset()
        train_dataset = TrainDataset(task_dataset=org_dataset)
        test_dataset = TestDataset(task_dataset=org_dataset)
        feature_extractor = FitFeatureModel(task_dataset=train_dataset)
        train_feature = ApplyFeatureModel(
            task_dataset=train_dataset, task_feature_extractor=feature_extractor
        )
        test_feature = ApplyFeatureModel(
            task_dataset=test_dataset, task_feature_extractor=feature_extractor
        )

        out = {"train": train_feature, "test": test_feature}

        return out

    def run(self) -> None:
        print("INFO: Running wrokflow...")
        self.dump(self.load())

