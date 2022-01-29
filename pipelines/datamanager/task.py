from typing import Any
import gokart
import pandas as pd
from .core._datamanager import _build_train_dataset, _build_test_dataset


class BuildTrainDataset(gokart.TaskOnKart):
    """
    Building training dataset
    """

    def requires(self) -> Any:
        return None

    def run(self) -> None:

        dataset: pd.DataFrame = _build_train_dataset()
        self.dump(dataset)


class BuildTestDataset(gokart.TaskOnKart):
    """
    Building test dataset
    """

    def requires(self) -> Any:
        return None

    def run(self) -> None:

        dataset: pd.DataFrame = _build_test_dataset()
        self.dump(dataset)
