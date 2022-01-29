from typing import List, Optional, Type, Tuple, Any
import luigi
from pathlib import Path
import pandas as pd
import pickle

from .datamanager import BuildDataset
from .features import BuildFeatures


class MainWorkflow(luigi.Task):
    """
    Workflow
    """

    def requires(self) -> Any:
        return {
            "feat1": BuildFeatures(
                task_dataset=BuildDataset(),
                param_model_name="Standardization",
                param_column_names=["sepal length (cm)"],
            ),
            "feat2": BuildFeatures(
                task_dataset=BuildDataset(),
                param_model_name="Standardization",
                param_column_names=["sepal width (cm)"],
            ),
        }

    def output(self) -> luigi.LocalTarget:
        filename = self.__class__.__name__ + ".pkl"
        _path = Path(__file__).parent / "../resources" / filename
        return luigi.LocalTarget(path=_path, format=luigi.format.Nop)

    def run(self) -> None:

        with self.input()["feat1"].open("rb") as input:
            dataset: pd.DataFrame = pickle.load(input)
            with self.output().open("wb") as output:
                pickle.dump(dataset, output, protocol=4)
