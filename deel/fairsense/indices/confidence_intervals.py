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
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from deel.fairsense.utils.dataclasses import IndicesInput
from deel.fairsense.utils.dataclasses import IndicesOutput


def with_confidence_intervals(n_splits=31, shuffle=False, random_state=None):
    """
    Function decorator that allows to compute confidence intervals using the naive
    method. The input data is split in n_splits and for each split indices are
    computed.

    Warnings:
            No correction if applied on the output (small number of split will lead
            to overconfident intervals and a large number of split will lead to a
            large variance due to the lack of data).

    This function must be applied on one of the indices computation function from the
    indices module.

    Args:
        n_splits: positive integer : number of split.
        shuffle:  Whether to shuffle the data before splitting into batches. Note that
            the samples within each split will not be shuffled.
        random_state: When `shuffle` is True, `random_state` affects the ordering of
            the indices, which controls the randomness of each fold. Otherwise, this
            parameter has no effect. Pass an int for reproducible output across
            multiple function calls.

    Returns:
        the original indice computation function enriched to compute confidence
        intervals.

    """

    kf = KFold(n_splits, shuffle=shuffle, random_state=random_state)

    def confidence_computation_fct(function):
        def call_function(inputs: IndicesInput, *args, **kwargs):
            # get full inputs
            x = inputs.x
            y = inputs.y_true
            fold_results = []
            # repeat indices computation on each fold
            for _, split in tqdm(kf.split(x, y), total=n_splits, ncols=80):
                # build input for the fold
                x_fold = x.iloc[split]
                y_fold = y.iloc[split] if y is not None else None
                fold_inputs = IndicesInput(
                    model=inputs.model,
                    x=x_fold,
                    y_true=y_fold,
                    variable_groups=inputs.variable_groups,
                    objective=inputs.objective,
                )
                # compute the result for the fold
                fold_results.append(function(fold_inputs, *args, **kwargs))
            # merge results to compute values and confidence intervals
            fvalues = [f.values for f in fold_results]
            runs = pd.concat(fvalues)
            return IndicesOutput(runs)

        return call_function

    return confidence_computation_fct
