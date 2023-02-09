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
from functools import partial

import numpy as np

from deel.fairsense.utils.dataclasses import IndicesOutput

"""
This module contains functions used to visualize the outputs of the indices in text 
format. The outputs of the indices are epresented as 
:mod:`.utils.dataclasses.IndicesOutput`.
"""


def format_with_intervals(indices_outputs: IndicesOutput, quantile: float = 0.05):
    """
    Pretty print the indices table with confidence intervals. Note that the intervals
    are displayed even if the indices are computed without confidence intervals. See
    :mod:`fairsense.indices.confidence_intervals` for more information.

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
