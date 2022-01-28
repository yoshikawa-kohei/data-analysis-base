from typing import List, Optional, Type, Tuple
import luigi
from pathlib import Path
import pandas as pd
import pickle

from .data import BuildDataset


class MainWorkflow(luigi.Task):
    """
    Workflow
    """

    def requires(self) -> Optional[Type[luigi.Task]]:
        return BuildDataset()

    def output(self) -> luigi.LocalTarget:
        __filename = self.__class__.__name__ + ".pkl"
        __path = Path(__file__).parent / "../resources" / __filename
        return luigi.LocalTarget(path=__path, format=luigi.format.Nop)

    def run(self) -> None:

        with self.input().open("r") as input:
            X: pd.DataFrame = pickle.load(input)
            with self.output().open("w") as output:
                output.write(pickle.dumps(X, protocol=4))
