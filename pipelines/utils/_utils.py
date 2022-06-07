from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd


class MemoryReducer:
    dtype_table: dict[str, list[Any]]
    n_jobs: int

    def __init__(
        self, dtype_table: Optional[dict[str, list[Any]]] = None, n_jobs: int = 2
    ) -> None:
        if dtype_table is None:
            self.dtype_table = {
                "int": [np.int8, np.int16, np.int32, np.int64],
                "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                "float": [np.float16, np.float32, np.float64],
            }
        else:
            self.dtype_table = dtype_table

        self.n_jobs = n_jobs

    def _type_candidates(self, dtype_key: str):
        for dtype in self.dtype_table[dtype_key]:
            if dtype_key == "int" or dtype_key == "uint":
                yield (dtype, np.iinfo(dtype))
            else:
                yield (dtype, np.finfo(dtype))

    def reduce(self, df: pd.DataFrame, verbose: bool = True):
        if verbose:
            print("====== Start Memory Reducer ======")

        columns: pd.Index = df.columns

        start_memory = df.memory_usage().sum() / 1024 ** 2

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executer:
            outputs = executer.map(
                partial(self._reduce, verbose=verbose), [df[col] for col in columns],
            )
        result = pd.concat([result for result in outputs], axis=1)

        end_memory = result.memory_usage().sum() / 1024 ** 2

        if verbose:
            print(
                "[Info]: memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                    end_memory, 100 * (start_memory - end_memory) / start_memory
                )
            )
            print("====== End Memory Reducer ======")

        return result

    def _reduce(self, input: pd.Series, verbose: bool) -> pd.Series:
        print(f"[Info]: reducing the column of '{input.name}'...")
        # skip NaNs
        if input.isnull().any():
            if verbose:
                print(f"[Info]: '{input.name}' has NaNs - Skip...")
            return input

        # detect kind of type
        org_dtype: np.dtype = input.dtype
        dtype_key: str
        if np.issubdtype(org_dtype, np.integer):
            dtype_key = "int" if input.min() < 0 else "uint"
        elif np.issubdtype(org_dtype, np.floating):
            dtype_key = "float"
        else:
            if verbose:
                print(f"[Info]: the dtype of '{input.name}' is '{org_dtype}' - Skip...")
            return input

        # find right candidate
        input_max = input.max()
        input_min = input.min()
        for dtype, dtype_info in self._type_candidates(dtype_key):
            if input_max <= dtype_info.max and input_min >= dtype_info.min:
                if verbose:
                    print(
                        f"[Info]: convert '{input.name}' from '{str(org_dtype)}' to '{str(dtype)}'"
                    )
                return input.astype(dtype)

        # reaching this code is bad. Probably there are inf, or other high numbs
        print(
            f"""[Warning]: {input.name}
            doesn't fit the grid with \nmax: {input.max()}
            and \nmin: {input.min()}"""
        )
        print("Dropping it..")


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "name": ["Walter", "David", "Jamie", "Kendra", "Zoey"],
            "age": [28, 31, 54, 44, 51],
        }
    )

    reducer = MemoryReducer()
    reducer.reduce(df)
