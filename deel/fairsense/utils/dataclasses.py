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
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from pandas import DataFrame


class IndicesInput:
    def __init__(
        self,
        model: Optional[Callable] = None,
        x: Optional[DataFrame] = None,
        y_true: Optional[DataFrame] = None,
        objective: Callable = None,
        variable_groups: List[List[str]] = None,
    ):
        """
        Build an IndiceInput object.

        Args:
            model: function that can be applied on x, that return a series with
            same shape as y_true.
            x: a dataframe containing the samples to analyse.
            y_true: a dataframe containing the labels of the samples (in the same
                order)
            objective: one of the target from the utils.fairness_objective module.
            variable_groups: list of list, containing the name of the columns that
                should be grouped.
        """

        self.model = model
        self._variable_groups = variable_groups
        self._x = x
        self._x.columns = [str(c) for c in x.columns]
        self._y_true = y_true
        self.objective = objective

    @property
    def x(self):
        # indice_input.x returns a copy of the data
        return self._x.copy()

    def compute_objective(self, x=None):
        """
        Compute the objective, using available data.
        When objective is y_true, y_true is returned, when objective is y_pred,
        the model is applied on x, and other objective compute the difference between
        y_true and y_pred.

        Args:
            x: the sample to compute the objective on. When None, `self.x` is used.

        Returns:
            the value of the objective.

        """
        return self.objective(self, x)

    @property
    def y_true(self):
        return self._y_true.copy()

    @y_true.setter
    def y_true(self, _y):
        # this setter ensures that y_true is a dataframe and not a series
        if _y is None:
            self._y_true = None
        else:
            if len(_y.shape) < 2:
                _y = np.expand_dims(_y, -1)
                self._y_true = DataFrame(_y, columns=["outputs"])
            elif isinstance(_y, DataFrame):
                self._y_true = _y
            else:
                self._y_true = DataFrame(_y, columns=["outputs"])

    @property
    def variable_groups(self):
        if self._variable_groups is None:
            return [[str(var)] for var in self._x.columns]
        else:
            return self._variable_groups

    @property
    def merged_groups(self):
        return [x[0].split("=")[0] for x in self.variable_groups]


class IndicesOutput:
    def __init__(self, values: DataFrame):
        """
        Encapsulate the results of the analysis. Every function from the indices
        module returns an object of this type.
        This object override the `+` operator allorw to combine result more easily.

        Args:
            values: a DataFrame containing the values of the indices. When confidence
            intervals are enabled this dataframe contains the results of each split.
        """
        self.runs: DataFrame = values  # 2D dataframe: lines=variable groups

    @property
    def values(self):
        return self.runs.groupby(level=0).median().clip(0.0, 1.0)

    def __add__(self, other):
        # indices must be computed on same groups
        assert other.runs.shape[0] == self.runs.shape[0]
        new_values = self.runs.copy()
        new_values[other.runs.columns] = other.runs
        return IndicesOutput(new_values)
