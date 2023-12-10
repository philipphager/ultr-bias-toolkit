import logging

import pandas as pd

from ultr_bias_toolkit.bias.intervention_harvesting.util import build_intervention_sets
from ultr_bias_toolkit.util.assertions import assert_columns_in_df


logger = logging.getLogger(__name__)


class AdjacentChainEstimator:
    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
    ) -> pd.DataFrame:
        logger.info(f"Position bias between adjacent/neighboring ranks")
        assert_columns_in_df(df, ["position", query_col, doc_col, "click"])

        df = build_intervention_sets(df, query_col, doc_col)
        # Filter interventions between adjacent pairs, prepend exam=1.0 for position 1:
        pos_1_df = df[(df.position_0 == 1) & (df.position_1 == 1)]
        adjacent_pair_df = df[df.position_1 == df.position_0 + 1]
        adjacent_pair_df = adjacent_pair_df.sort_values(["position_0", "position_1"])
        df = pd.concat([pos_1_df, adjacent_pair_df])

        # Compute click ratio between neighboring ranks:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df["examination"] = df.examination.cumprod()

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]].reset_index(drop=True)
