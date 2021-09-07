from typing import Callable, Optional, List
from pandas import DataFrame
import numpy as np


class IndicesInput:
    def __init__(
        self,
        model: Optional[Callable] = None,
        x: Optional[DataFrame] = None,
        y: Optional[DataFrame] = None,
        variable_groups: List[List[str]] = None,
    ):

        self.model = model
        self._variable_groups = variable_groups
        self._x = x
        self._x.columns = [str(c) for c in x.columns]
        self._y = y

    @property
    def x(self):
        # indice_input.x returns a copy of the data
        return self._x.copy()

    @property
    def y(self):
        # get y if a sample is given or compute it with the model
        if self._y is not None:
            return self._y
        elif self._x is not None and self.model is not None:
            return self.model(self.x.values)
        else:
            raise RuntimeError(
                "y must be set, or x and a model must be given in "
                "order to acces y attribute"
            )

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
        return self.runs.groupby(level=0).median().clip(0., 1.)

    def __add__(self, other):
        # indices must be computed on same groups
        assert other.runs.shape[0] == self.runs.shape[0]
        new_values = self.runs.copy()
        new_values[other.runs.columns] = other.runs
        return IndicesOutput(new_values)