from typing import Any, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .base import FeatureModelBase

class Standardization(FeatureModelBase):
    def __init__(
        self,
        column_names: List[str],
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.__settings: Dict[str, Any] = {
            "copy": copy,
            "with_mean": with_mean,
            "with_std": with_std,
        }
        self.__engine: Any = StandardScaler(**self.__settings)
        self.__engine_name: str = "Standardization"
        self.__column_names: List[str] = column_names

    @property
    def settings(self) -> Dict[str, Any]:
        return self.__settings

    @property
    def engine_name(self) -> str:
        return self.__engine_name

    def fit(self, data: pd.DataFrame) -> Any:
        self.__engine.fit(data[self.__column_names])
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.__engine.transform(data[self.__column_names]),
            columns=self.__column_names,
        )
