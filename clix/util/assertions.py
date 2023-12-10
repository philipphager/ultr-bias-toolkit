import pandas as pd
from typing import List


def assert_columns_in_df(df: pd.DataFrame, columns: List[str]):
    for c in columns:
        assert (
            c in df.columns
        ), f"Expected DataFrame to contain: {columns}, found: {df.columns}"
