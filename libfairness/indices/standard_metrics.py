from warnings import warn

from libfairness.data_management import utils
from libfairness.fairness_problem import FairnessProblem

import numpy as np
import pandas as pd

# ---- DISPARATE IMPACT ----
from libfairness.utils.dataclasses import IndicesInput, IndicesOutput


def disparate_impact(index_input: IndicesInput) -> IndicesOutput:
    df = index_input.x
    y = index_input.y
    dis = []
    for group in index_input.variable_groups:
        group_output = []
        for var in group:
            group_output.append(disparate_impact_single_variable(df[var], y))
        dis.append(np.mean(group_output))
    data = np.expand_dims(np.array(dis), axis=-1)
    index = index_input.merged_groups
    results = pd.DataFrame(data=data, columns=["DI"], index=index)
    return IndicesOutput(results)


def disparate_impact_single_variable(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame(np.expand_dims(x, -1), columns=["X"])
    df["outputs"] = y
    succes_probs = df.groupby("X")["outputs"].mean()
    if len(succes_probs) > 2:
        warn(f"non binary variable {x.name} encountered in DI, replacing with nan.")
        return np.nan
    di = succes_probs.min() / succes_probs.max()
    return di