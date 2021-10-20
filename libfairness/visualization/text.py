import numpy as np
from functools import partial
from libfairness.utils.dataclasses import IndicesOutput

"""
This module contains functions used to visualize the outputs of the indices in text 
format. The outputs of the indices are epresented as 
:mod:`.utils.dataclasses.IndicesOutput`.
"""


def format_with_intervals(indices_outputs: IndicesOutput, quantile: float = 0.05):
    f"""
    Pretty print the indices table with confidence intervals. Note that the intervals
    are displayed even if the indices are computed without confidence intervals. See
    :mod:`libfairness.indices.confidence_intervals` for more information.

    Args:
        indices_outputs (IndicesOutput): computed indices
        quantile (float): quantile used to compute confidence intervals. Values must
        be in [0., 0.5].

    Returns: the table with indices properly displayed. Note that the table values
        are now string and not float.

    """
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
