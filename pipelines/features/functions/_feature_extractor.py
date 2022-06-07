from lib2to3.pytree import Base
from typing import Any, Optional

import pandas as pd

from .models import BaseFeature, Standardization


class FeatureExtractor:
    __models: list[BaseFeature]

    def __init__(self, models: list[BaseFeature]) -> None:
        self.__models = models

    def fit(self, data: pd.DataFrame) -> None:
        for model in self.__models:
            model.fit(data)

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        outputs: list[pd.DataFrame] = []
        outputs.extend([model.apply(data) for model in self.__models])

        return pd.concat(outputs, axis="columns") if outputs else pd.DataFrame()

