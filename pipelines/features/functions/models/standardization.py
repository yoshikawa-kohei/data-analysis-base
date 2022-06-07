from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ._feature import BaseFeature


class Standardization(BaseFeature):
    def __init__(
        self,
        column_names: list[str],
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.__settings: dict[str, Any] = {
            "column_names": column_names,
            "copy": copy,
            "with_mean": with_mean,
            "with_std": with_std,
        }
        self.__model = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

    @property
    def settings(self) -> dict[str, Any]:
        return self.__settings

    def fit(self, data: pd.DataFrame) -> Any:
        self.__model.fit(data[self.__settings["column_names"]])
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.__model.transform(data[self.__settings["column_names"]]),
            columns=self.__settings["column_names"],
        )
