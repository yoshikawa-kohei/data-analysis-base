from typing import List, Any, Optional
from unicodedata import name
import pandas as pd
from .base import FeatureModelBase
from .standardization import Standardization


class FeatureModel(FeatureModelBase):
    def __init__(
        self,
        models: Optional[List[Any]] = None,
        rule_based_models: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(model=models, name="Parent Feature Model")
        self._rule_based_model: Optional[List[Any]] = rule_based_models

    def fit(self, data: pd.DataFrame) -> Any:
        if self._model is None:
            print(
                "INFO: There are no registered feature models, so no fitting will be performed."
            )
            return self

        for model in self._model:
            model.fit(data)
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        outputs: List[pd.DataFrame] = []
        if self._model is None:
            print(
                "INFO: There are no registered feature models, so no applying will be performed."
            )
        else:
            outputs.extend([model.apply(data) for model in self._model])

        if self._rule_based_model is None:
            print(
                "INFO: There are no registered rule-based feature models, so no applying will be performed."
            )
        else:
            outputs.extend([model.apply(data) for model in self._rule_based_model])

        return pd.concat(outputs, axis="columns") if outputs else pd.DataFrame()


def _train_feature_model(data: pd.DataFrame) -> Any:
    # child models
    std_model: Any = Standardization(column_names=["sepal width (cm)"])

    # parent model
    model = FeatureModel(models=[std_model])
    model.fit(data)

    return model
