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
"""
This module contains factory functions that allow to build more easily IndicesInput
objects.
"""
from typing import Callable
from typing import Optional
from typing import Union

import pandas as pd

from deel.fairsense.utils.dataclasses import IndicesInput


def from_pandas(
    dataframe: pd.DataFrame,
    y: Union[str, pd.DataFrame, pd.Series, None],
    model: Optional[Callable] = None,
    target: Callable = None,
) -> IndicesInput:
    """
    Builds IndicesInput from pandas dataframe.

    Args:
        dataframe: DataFrame containing the samples to analyse.
        y: Union[str, pd.DataFrame, pd.Series, None] : when str, refers to the name
            of the columns containing the labels. Must be present in dataframe. When
            pd.DataFrame or pd.Series the label are provided in the same order as in
            dataframe. When None, no labels are provided.
        model: function that can be applied on dataframe, that return a series with
            same shape as y.
        target: one of the target from the utils.fairness_objective module.

    Returns:
        an IndicesInput object that can be used to compute sensitivity indices.

    """
    cols = set(dataframe.columns)
    if y is None:
        assert model is not None, "model must be defined when target is None"
        x = dataframe
        y = None
    elif isinstance(y, str):
        x = dataframe[cols - {y}]
        y = dataframe[y]
    elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        x = dataframe
        y = pd.DataFrame(y)
    else:
        raise RuntimeError("type of target must be Dataframe, Series, str or None")
    return IndicesInput(x=x, y_true=y, model=model, objective=target)


def from_numpy(x, y, feature_names=None, model=None, target=None) -> IndicesInput:
    """
    Builds IndicesInput from numpy array.

    Args:
        x: numpy array containing the samples to analyse.
        y: numpy array containing the labels. Can be None if no labels are provided.
        feature_names: a list of str containing the features name of x. When None
            features are named with numbers.
        model: function that can be applied on dataframe, that return an series with
            same shape as y.
        target: one of the target from the utils.fairness_objective module.

    Returns:
        an IndicesInput object that can be used to compute sensitivity indices.

    """
    df = pd.DataFrame(x, columns=feature_names)
    # build dataframe
    y = pd.DataFrame(y, columns=["target"])
    return from_pandas(dataframe=df, y=y, model=model, target=target)
