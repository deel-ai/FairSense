from typing import Callable, Optional, List
from pandas import DataFrame
import numpy as np


class IndicesInput:
    def __init__(
        self,
        model: Optional[Callable] = None,
        x: Optional[DataFrame] = None,
        y: Optional[DataFrame] = None,
        target: Callable = None,
        variable_groups: List[List[str]] = None,
    ):

        self.model = model
        self._variable_groups = variable_groups
        self._x = x
        self._x.columns = [str(c) for c in x.columns]
        self._y = y
        self._target = target

    @property
    def x(self):
        # indice_input.x returns a copy of the data
        return self._x.copy()

    def get_target(self, x=None):
        return self._target(self, x)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _y):
        # this setter ensures that y is a dataframe and not a series
        if _y is None:
            self._y = None
        else:
            if len(_y.shape) < 2:
                _y = np.expand_dims(_y, -1)
                self._y = DataFrame(_y, columns=["outputs"])
            elif isinstance(_y, DataFrame):
                self._y = _y
            else:
                self._y = DataFrame(_y, columns=["outputs"])

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
