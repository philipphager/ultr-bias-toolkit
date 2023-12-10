# Personal toolkit for bias estimation in unbiased learning to rank

## Installation
```
pip install ultr-bias-toolkit
```

## Offline bias estimation methods
We implement multiple offline position bias estimation methods, including [three intervention harvesting](https://arxiv.org/abs/1812.05161) approaches:

```
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator, AdjacentChainEstimator, AllPairsEstimator

estimators = {
    "CTR Rate": NaiveCtrEstimator(),
    "Pivot One": PivotEstimator(pivot_rank=1),
    "Adjacent Chain": AdjacentChainEstimator(),
    "Global All Pairs": AllPairsEstimator(),
}
examination_dfs = []

for name, estimator in estimators.items():
    examination_df = estimator(df)
    examination_df["estimator"] = name
    examination_dfs.append(examination_df)

examination_df = pd.concat(examination_dfs)
examination_df.head()
```

Example on the [Baidu ULTR dataset](https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-base-12L):
![baidu](https://github.com/philipphager/ultr-bias-toolkit/assets/9155371/d8cca083-9722-42eb-a051-8479153d809c)
