from typing import List, Optional, Type, Tuple
import luigi
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

class BuildDataset(luigi.Task):
    """
    Building dataset
    """

    def requires(self) -> Optional[Type[luigi.Task]]:
        return None

    def output(self) -> luigi.LocalTarget:
        __filename = self.__class__.__name__ + ".pkl"
        __path = Path(__file__).parent / "../../resources" / __filename
        return luigi.LocalTarget(path=__path, format=luigi.format.Nop)

    def run(self) -> None:
        from sklearn.datasets import load_iris

        X: pd.DataFrame = load_iris(as_frame=True)

        with self.output().open('w') as output:
            output.write(pickle.dumps(X, protocol=4))
