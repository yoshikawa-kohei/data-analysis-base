
from typing import Any, Dict, List
import pandas as pd

from .standardization import Standardization


def strclass(class_name: str) -> Any:
    return globals()[class_name]()

def _build_features() -> pd.DataFrame:
    from sklearn.utils import Bunch
    from sklearn.datasets import load_iris

    iris: Bunch = load_iris(as_frame=True)
    source: pd.DataFrame = iris.data
    target: pd.DataFrame = iris.target

    dataset: pd.DataFrame = pd.concat([target, source], axis="columns")

    return dataset


def _train_features(data: pd.DataFrame, model_name: str, column_names: List[str]) -> Any:
    model: Any = strclass(model_name)
    model.fit(data=data, column_names=list(column_names))
    return model
