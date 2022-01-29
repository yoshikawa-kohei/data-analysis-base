from typing import List, Optional, Type, Tuple, Any
import luigi
from pathlib import Path
import pandas as pd
import pickle

from .core._datamanager import _build_dataset

from ..utils import RESOURCES_PATH


class BuildDataset(luigi.Task):
    """
    Building dataset
    """

    def requires(self) -> Optional[Type[luigi.Task]]:
        return None

    def output(self) -> luigi.LocalTarget:
        filename = self.__class__.__name__ + ".pkl"
        return luigi.LocalTarget(
            path=RESOURCES_PATH / filename, format=luigi.format.Nop
        )

    def run(self) -> None:

        dataset: pd.DataFrame = _build_dataset()

        with self.output().open("wb") as output:
            pickle.dump(dataset, output, protocol=4)
