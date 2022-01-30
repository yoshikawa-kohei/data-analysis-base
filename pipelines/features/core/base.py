import imp
from typing import Any, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import abstractmethod, ABCMeta

class FeatureModelBase(metaclass=ABCMeta):
    def __init__(
        self,
    ) -> None:
        self.__engine: Any
        self.__engine_name: str

    @property
    def engine_name(self) -> str:
        return self.__engine_name

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
