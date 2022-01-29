from typing import List, Optional, Type, Tuple, Any, Dict
import luigi
from luigi.util import inherits
from pathlib import Path
import pandas as pd
import pickle

from ..datamanager import BuildDataset
from .core._features import _build_features, _train_features

from ..utils import RESOURCES_PATH, strhash


class TrainFeatureModel(luigi.Task):
    task_dataset = luigi.TaskParameter()
    param_model_name = luigi.Parameter()
    param_column_names = luigi.ListParameter()

    def requires(self) -> Any:
        return self.task_dataset

    def output(self) -> luigi.LocalTarget:
        hash: str = strhash(self.param_model_name + "".join(self.param_column_names))
        filename = self.__class__.__name__ + "_" + hash + ".pkl"
        return luigi.LocalTarget(
            path=RESOURCES_PATH / filename, format=luigi.format.Nop
        )

    def run(self) -> None:
        with self.input().open("rb") as input:
            dataset: pd.DataFrame = pickle.load(input)
            build_features_model = _train_features(
                dataset, self.param_model_name, self.param_column_names
            )

        with self.output().open("wb") as output:
            pickle.dump(build_features_model, output, protocol=4)


class BuildFeatures(luigi.Task):
    """
    Building features
    """

    task_dataset = luigi.TaskParameter()
    param_model_name = luigi.Parameter()
    param_column_names = luigi.ListParameter()

    def requires(self) -> Any:
        return {
            "dataset": self.task_dataset,
            "model": TrainFeatureModel(
                task_dataset=self.task_dataset,
                param_model_name=self.param_model_name,
                param_column_names=self.param_column_names,
            ),
        }

    def output(self) -> luigi.LocalTarget:
        hash: str = strhash(
            self.task_dataset.__class__.__name__
            + self.param_model_name
            + "".join(self.param_column_names)
        )
        filename = self.__class__.__name__ + "_" + hash + ".pkl"
        return luigi.LocalTarget(
            path=RESOURCES_PATH / filename, format=luigi.format.Nop
        )

    def run(self) -> None:

        with self.input()["dataset"].open("rb") as input_dataset, self.input()[
            "model"
        ].open("rb") as input_model:
            dataset: pd.DataFrame = pickle.load(input_dataset)
            model: Any = pickle.load(input_model)

            features: pd.DataFrame = model.apply(dataset)

        with self.output().open("wb") as output:
            pickle.dump(features, output, protocol=4)
