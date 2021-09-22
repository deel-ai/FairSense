from typing import List
import pandas as pd
from libfairness.utils.dataclasses import IndicesInput


def one_hot_encode(
    indices_input: IndicesInput, categorical_variables: List[str] = None
) -> IndicesInput:
    x = indices_input.x
    orig_var_groups = indices_input.variable_groups
    out_x = pd.get_dummies(
        x,
        prefix=categorical_variables,
        prefix_sep="=",
        dummy_na=False,
        drop_first=True,
        columns=categorical_variables,
    )
    out_var_groups = []
    # read the original groups
    for group in orig_var_groups:
        # iterate through the variable names in the group
        new_group = []
        for c in group:
            # in the group seek for the new variables names
            if c in out_x.columns:
                # variable has not been one hot encoded yet
                new_group += [c]
            else:
                # variable has been one hot encoded, find the new variable names and
                # add it to the group
                new_group += list(
                    filter(lambda cname: cname.startswith(c), out_x.columns)
                )
        # group is finished add the new group to groups
        out_var_groups.append(new_group)
    return IndicesInput(
        x=out_x,
        y_true=indices_input.y_true,
        model=indices_input.model,
        variable_groups=out_var_groups,
        objective=indices_input._objective,
    )
