from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import abstractmethod, ABCMeta


class FeatureModelBase(metaclass=ABCMeta):
    def __init__(self, model: Optional[Any] = None, name: str = "") -> None:
        self._model: Any = model
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
