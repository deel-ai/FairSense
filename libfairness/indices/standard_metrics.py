from warnings import warn
import numpy as np
import pandas as pd

# ---- DISPARATE IMPACT ----
from libfairness.utils.dataclasses import IndicesInput, IndicesOutput


def disparate_impact(index_input: IndicesInput) -> IndicesOutput:
    df = index_input.x
    y = index_input.compute_objective()
    df["outputs"] = y
    dis = []
    for group in index_input.variable_groups:
        group_output = []
        for var in group:
            group_output.append(_disparate_impact_single_variable(df, var))
        dis.append(np.mean(group_output))
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
