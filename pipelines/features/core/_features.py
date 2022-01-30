from typing import List, Any, Optional
import pandas as pd
from .standardization import Standardization


class FeatureModel:
    def __init__(
        self, model: Optional[List[Any]] = None, rule_based_model: Optional[List[Any]] = None
    ) -> None:
        self.model: Optional[List[Any]] = model
        self.rule_based_model: Optional[List[Any]] = rule_based_model

    def fit(self, data: pd.DataFrame) -> Any:
        if self.model is None:
            print("INFO: There are no registered feature models, so no fitting will be performed.")
            return self

        for model in self.model:
            model.fit(data)
        return self

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        outputs: List[pd.DataFrame] = []
        if self.model is None:
            print("INFO: There are no registered feature models, so no applying will be performed.")
        else:
            outputs.extend([model.apply(data) for model in self.model])

        if self.rule_based_model is None:
            print("INFO: There are no registered rule-based feature models, so no applying will be performed.")
        else:
            outputs.extend([model.apply(data) for model in self.rule_based_model])

        return pd.concat(outputs, axis="columns") if outputs else pd.DataFrame()


def _train_feature_model(data: pd.DataFrame) -> Any:
    # child models
    std_model: Any = Standardization(column_names=["sepal width (cm)"])

    # parent model
    model = FeatureModel(model=[std_model])
    model.fit(data)

    return model
