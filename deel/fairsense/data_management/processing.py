# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import List

import pandas as pd

from deel.fairsense.utils.dataclasses import IndicesInput


def one_hot_encode(
    indices_input: IndicesInput, categorical_variables: List[str] = None
) -> IndicesInput:
    """
    Performs one-hot encoding on the specified categorical variables. Variable groups
    are updated accordingly. Newly created variables are named:
    `original_feature_name=value`

    Args:
        indices_input: IndiceInput object containing the data.
        categorical_variables: name of the variable that should be encoded.

    Returns:
        the updated IndicesInput.

    """
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
                    filter(lambda cname: cname.startswith(c + "="), out_x.columns)
                )
        # group is finished add the new group to groups
        out_var_groups.append(new_group)
    return IndicesInput(
        x=out_x,
        y_true=indices_input.y_true,
        model=indices_input.model,
        variable_groups=out_var_groups,
        objective=indices_input.objective,
    )
