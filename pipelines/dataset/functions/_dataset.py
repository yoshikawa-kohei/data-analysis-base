import pandas as pd


def _build_train_dataset() -> pd.DataFrame:
    from sklearn.utils import Bunch
    from sklearn.datasets import load_iris

    iris: Bunch = load_iris(as_frame=True)
    source: pd.DataFrame = iris.data
    target: pd.DataFrame = iris.target

    dataset: pd.DataFrame = pd.concat([target, source], axis="columns")

    return dataset


def _build_test_dataset() -> pd.DataFrame:
    from sklearn.utils import Bunch
    from sklearn.datasets import load_iris

    iris: Bunch = load_iris(as_frame=True)
    source: pd.DataFrame = iris.data

    return source
