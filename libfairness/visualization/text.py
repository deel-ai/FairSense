import numpy as np
from functools import partial
from libfairness.utils.dataclasses import IndicesOutput


def format_with_intervals(indices_outputs: IndicesOutput, quantile: int = 0.05):
    means = indices_outputs.runs.groupby(level=0).median().clip(0.0, 1.0)
    low = (
        indices_outputs.runs.groupby(level=0)
        .aggregate(partial(np.quantile, q=quantile))
        .clip(0.0, 1.0)
    )
    high = (
        indices_outputs.runs.groupby(level=0)
        .aggregate(partial(np.quantile, q=1 - quantile))
        .clip(0.0, 1.0)
    )
    table = means.copy()
    for index in means.columns:
        table[index] = np.vectorize(
            lambda index_val, index_inf_val, index_sup_val: "%.2f [%.2f, %.2f]"
            % (index_val, index_inf_val, index_sup_val)
        )(means[index], low[index], high[index])
    return table
