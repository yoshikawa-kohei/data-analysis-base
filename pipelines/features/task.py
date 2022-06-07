from typing import Any

import gokart
import luigi
import pandas as pd

from .functions import FeatureExtractor
from .functions.models import BaseFeature, Standardization


class FitFeatureModel(gokart.TaskOnKart):
    task_dataset = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return self.task_dataset

    def run(self) -> None:
        dataset: pd.DataFrame = self.load()

        extractor = FeatureExtractor(models=self._feature_models())
        extractor.fit(dataset)

        self.dump(extractor)

    def _feature_models(self) -> list[BaseFeature]:
        # child models
        models: list[BaseFeature] = [
            Standardization(column_names=["sepal width (cm)"]),
            Standardization(column_names=["sepal length (cm)"]),
        ]

        return models


class ApplyFeatureModel(gokart.TaskOnKart):
    task_dataset = gokart.TaskInstanceParameter()
    task_feature_extractor = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return {"dataset": self.task_dataset, "model": self.task_feature_extractor}

    def run(self) -> None:
        dataset: pd.DataFrame = self.load("dataset")
        model: Any = self.load("model")

        features: pd.DataFrame = model.apply(dataset)

        self.dump(features)
