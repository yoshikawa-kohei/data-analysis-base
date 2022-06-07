from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFeature(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
