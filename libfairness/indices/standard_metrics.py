from warnings import warn
import numpy as np
import pandas as pd
from libfairness.utils.dataclasses import IndicesInput, IndicesOutput


def disparate_impact(index_input: IndicesInput, group_reduction=np.mean) -> IndicesOutput:
    """
    Compute the disparate impact.

    Warning:
        disparate impact/equality of odds can only be computed on classification
        problems, and on categorical variables. Continuous variables are dropped and
        output replaced by `np.nan`

    Note:
         When applied with `target=classification_error` this function compute the
         equality of odds.

    Args:
        index_input (IndicesInput): The fairness problem to study.
        group_reduction: the method used to compute the indices for a group of
            variables. By default the average of the values of each groups is applied.

    Returns:
        IndicesOutput object, containing the CVM indices, one line per variable group
        and one column for each index.

    """
    df = index_input.x
    y = index_input.compute_objective()
    df["outputs"] = y.values
    dis = []
    for group in index_input.variable_groups:
        group_output = []
        for var in group:
            group_output.append(_disparate_impact_single_variable(df, var))
        dis.append(group_reduction(group_output))
    data = np.expand_dims(np.array(dis), axis=-1)
    index = index_input.merged_groups
    results = pd.DataFrame(data=data, columns=["DI"], index=index)
    return IndicesOutput(results)


def _disparate_impact_single_variable(df: pd.Series, variable: str) -> float:
    succes_probs = df[[variable, "outputs"]].groupby(variable)["outputs"].mean()
    if len(succes_probs) > 2:
        # warn(f"non binary variable {variable} encountered in DI, replacing with nan.")
        return np.nan
    di = 1.0 - succes_probs.min() / (succes_probs.max() + 1e-7)
    return di
