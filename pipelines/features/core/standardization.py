
from typing import Any, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Standardization:
    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ) -> None:
        self.__settings: Dict[str, Any] = {
            "copy": copy,
            "with_mean": with_mean,
            "with_std": with_std,
        }
        self.__engine: Any = StandardScaler(**self.__settings)
        self.__engine_name: str = "Standardization"

    @property
    def settings(self) -> Dict[str, Any]:
        return self.__settings

    @property
    def engine_name(self) -> str:
        return self.__engine_name

    def fit(self, data: pd.DataFrame, column_names: List[str]) -> Any:
        self.__column_names = column_names
        self.__engine.fit(data[self.__column_names])
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.__engine.transform(data[self.__column_names]),
            columns=self.__column_names,
        )
