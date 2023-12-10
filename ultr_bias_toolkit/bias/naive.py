import pandas as pd

from ultr_bias_toolkit.bias.intervention_harvesting.util import normalize_bias
from ultr_bias_toolkit.util.assertions import assert_columns_in_df


class NaiveCtrEstimator:
    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
    ) -> pd.DataFrame:
        assert_columns_in_df(df, ["position", query_col, doc_col, "click"])

        df = df.groupby("position").agg(
            examination=("click", "mean")
        ).reset_index()

        df.examination = normalize_bias(df.examination)
        return df
