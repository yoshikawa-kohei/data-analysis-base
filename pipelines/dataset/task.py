from typing import Any

import gokart
import pandas as pd

from .functions import MemoryReducer


class BuildDataset(gokart.TaskOnKart):
    """
    Building dataset
    """

    def requires(self) -> Any:
        return None

    def run(self) -> None:

        df: pd.DataFrame = self._build_dataset()
        self.dump(df)

    def _build_dataset(self) -> pd.DataFrame:
        from sklearn.datasets import load_iris
        from sklearn.utils import Bunch

        iris: Bunch = load_iris(as_frame=True)
        source: pd.DataFrame = iris.data
        target: pd.DataFrame = iris.target

        df: pd.DataFrame = pd.concat([target, source], axis="columns")

        df = self._preprocess(df)

        reducer = MemoryReducer(n_jobs=2)
        result: pd.DataFrame = reducer.reduce(df=df, verbose=True)

        return result

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class TrainDataset(gokart.TaskOnKart):

    task_dataset = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return self.task_dataset

    def run(self) -> None:
        df: pd.DataFrame = self.load()
        self.dump(df)


class TestDataset(gokart.TaskOnKart):

    task_dataset = gokart.TaskInstanceParameter()

    def requires(self) -> Any:
        return self.task_dataset

    def run(self) -> None:
        df: pd.DataFrame = self.load()
        self.dump(df)
