import logging

import pandas as pd

from ultr_bias_toolkit.bias.intervention_harvesting.util import build_intervention_sets
from ultr_bias_toolkit.bias.intervention_harvesting.util import normalize_bias
from ultr_bias_toolkit.util.assertions import assert_columns_in_df

logger = logging.getLogger(__name__)


class PivotEstimator:
    def __init__(self, pivot_rank: int = 1):
        self.pivot_rank = pivot_rank

    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
    ) -> pd.DataFrame:
        logger.info(f"Position bias between rank k and pivot rank: {self.pivot_rank}")
        assert_columns_in_df(df, ["position", query_col, doc_col, "click"])

        df = build_intervention_sets(df, query_col, doc_col)
        # Filter interventions with pivot rank in first positions:
        df = df[df.position_0 == self.pivot_rank]
        # Computing CTR ratio between position k and the pivot rank:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df.examination = normalize_bias(df.examination)

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]]
