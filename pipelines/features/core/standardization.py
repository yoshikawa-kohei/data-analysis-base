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
        self._settings: Dict[str, Any] = {
            "copy": copy,
            "with_mean": with_mean,
            "with_std": with_std,
        }
        super().__init__(model=StandardScaler(**self._settings), name="Standardization")
        self._column_names: List[str] = column_names

    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings

    def fit(self, data: pd.DataFrame) -> Any:
        self._model.fit(data[self._column_names])
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=self._model.transform(data[self._column_names]),
            columns=self._column_names,
        )
