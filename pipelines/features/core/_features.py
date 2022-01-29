
from typing import Any
import pandas as pd
from .standardization import Standardization


def strclass(class_name: str) -> Any:
    return globals()[class_name]()


def _train_feature_model(data: pd.DataFrame) -> Any:
    model: Any = Standardization()
    model.fit(data=data, column_names=["sepal width (cm)"])
    return model
