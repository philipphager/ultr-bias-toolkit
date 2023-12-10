import pandas as pd


def build_intervention_sets(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df["no_click"] = 1 - df["click"]
    df = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=("click", "sum"),
            no_clicks=("no_click", "sum"),
            impressions=(doc_col, "count"),
        )
        .reset_index()
    )

    df["c"] = df["clicks"] / df["impressions"]
    df["not_c"] = df["no_clicks"] / df["impressions"]
    df = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])

    df = (
        df.groupby(["position_0", "position_1"])
        .agg(
            c_0=("c_0", "sum"),
            c_1=("c_1", "sum"),
            not_c_0=("not_c_0", "sum"),
            not_c_1=("not_c_1", "sum"),
        )
        .reset_index()
    )

    return df


def normalize_bias(examination: pd.Series) -> pd.Series:
    examination /= examination.values[0]
    return examination.fillna(0)
