from typing import Any
import luigi
import gokart
import pandas as pd
from .core._features import _train_feature_model


class TrainFeatureModel(gokart.TaskOnKart):
    task_dataset = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return self.task_dataset

    def run(self) -> None:
        dataset: pd.DataFrame = self.load()
        build_features_model = _train_feature_model(dataset)

        self.dump(build_features_model)


class ApplyFeatureModel(gokart.TaskOnKart):
    task_dataset = gokart.TaskInstanceParameter()
    task_feature_model = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return {"dataset": self.task_dataset, "model": self.task_feature_model}

    def run(self) -> None:
        dataset: pd.DataFrame = self.load("dataset")
        model: Any = self.load("model")

        features: pd.DataFrame = model.apply(dataset)

        self.dump(features)
